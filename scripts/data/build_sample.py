"""Select A61B utility patents and construct exact five-anniversary boundaries."""
from common import ROOT, connect, parquet, qp, receipt, setting, source


def main():
    connection = connect()
    dates = source("02_core/patent_dates.parquet")
    cpc = source("02_core/patent_cpc.parquet")
    primary = source("02_core/patent_primary_field.parquet")
    patents = source("01_stage_parquet/patent.parquet")
    observation_end = setting("study", "observation_end", "2025-12-30")
    connection.execute("CREATE VIEW dates AS SELECT * FROM read_parquet(" + qp(dates) + ")")
    connection.execute("CREATE VIEW patents AS SELECT * FROM read_parquet(" + qp(patents) + ")")
    for table in ("dates", "patents"):
        total, unique = connection.execute(f"SELECT count(*), count(DISTINCT patent_id) FROM {table}").fetchone()
        if total != unique:
            raise ValueError(f"{table} must have one row per patent_id")
    connection.execute("""CREATE TABLE candidate_groups AS
      SELECT p.patent_id,
        arg_min(p.cpc_group, struct_pack(seq:=coalesce(p.cpc_sequence,32767), grp:=p.cpc_group)) first_a61b_group,
        min(p.cpc_sequence) first_a61b_sequence, count(DISTINCT p.cpc_group) a61b_group_count
      FROM read_parquet(""" + qp(cpc) + """) p JOIN dates d USING(patent_id)
      WHERE p.cpc_subclass='A61B' AND d.grant_year BETWEEN 1995 AND 2020
      GROUP BY p.patent_id""")
    connection.execute("""CREATE TABLE candidates AS
      SELECT CAST(g.patent_id AS VARCHAR) patent_id,
        CAST(d.patent_date AS DATE) grant_date, CAST(d.filing_date AS DATE) filing_date,
        d.grant_year, d.filing_year, g.first_a61b_group, g.first_a61b_sequence, g.a61b_group_count,
        pf.cpc_subclass current_primary_cpc_subclass, pf.cpc_group current_primary_cpc_group,
        g.first_a61b_sequence=0 primary_sequence_is_a61b,
        p.withdrawn withdrawn_flag, p.patent_type,
        (d.grant_year BETWEEN 2005 AND 2020) is_focal_candidate,
        (d.grant_year BETWEEN 1995 AND 2004) is_history_warmup,
        CAST(CAST(d.patent_date AS DATE)+INTERVAL '5 YEARS' AS DATE) label_window_end,
        CAST(CAST(d.patent_date AS DATE)+INTERVAL '5 YEARS' AS DATE)<=CAST(""" + qp(observation_end) + """ AS DATE) mature_by_local_snapshot,
        CAST(""" + qp(observation_end) + """ AS DATE) local_snapshot_end
      FROM candidate_groups g JOIN dates d USING(patent_id)
      JOIN patents p USING(patent_id)
      LEFT JOIN read_parquet(""" + qp(primary) + """) pf USING(patent_id)
      WHERE p.patent_type='utility' AND coalesce(p.withdrawn,0)=0""")
    total, unique = connection.execute("SELECT count(*),count(DISTINCT patent_id) FROM candidates").fetchone()
    if total != unique:
        raise ValueError("Primary classification table created duplicate patent rows")
    parquet(connection, "SELECT * FROM candidates ORDER BY grant_date,patent_id", "data/a61b_candidates_1995_2020.parquet")
    counts = connection.execute("SELECT grant_year,count(*) patents FROM candidates GROUP BY grant_year ORDER BY grant_year").fetchdf()
    counts.to_csv(ROOT / "data/sample_counts_by_year.csv", index=False)
    receipt("checks/sample.json", [dates, cpc, primary, patents], patents=total,
            observation_end=observation_end,
            scope="1995–2004 historical reference and 2005–2020 focal grants; utility, non-withdrawn, CPC A61B")
    print(f"Prepared {total:,} sample records.")


if __name__ == "__main__":
    main()
