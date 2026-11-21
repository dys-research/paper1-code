"""Local first-appearance shares of distinct unigrams, bigrams and trigrams."""
from scripts.robustness.common import *
import duckdb
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def main():
    all_patents = panel(all_rows=True)
    text_path = source('text_panel', work_dir('data') / 'data/a61b_text_base_1995_2020.parquet')
    if not text_path.exists():
        raise FileNotFoundError(f'Prepared patent text panel is required: {text_path}')
    connection = duckdb.connect()
    connection.execute('SET threads=2')
    connection.execute("SET memory_limit='3GB'")
    connection.execute('SET temp_directory=?', [str(output('tmp/lexical'))])
    connection.execute('SET VARIABLE stops = ?', [sorted(ENGLISH_STOP_WORDS)])
    connection.register('dates', all_patents[['patent_id', 'grant_date']])
    path = str(text_path).replace("'", "''")
    connection.execute(f'''CREATE TEMP TABLE tokens AS SELECT d.patent_id,CAST(d.grant_date AS DATE) grant_date,
        list_filter(regexp_split_to_array(regexp_replace(lower(concat_ws(' ',
        t.patent_title,t.patent_abstract,t.independent_claim_text)), '[^a-z]', ' ', 'g'),'\\s+'),
        lambda word: len(word)>=3 AND NOT list_contains(getvariable('stops'),word)) words
        FROM read_parquet('{path}') t JOIN dates d USING(patent_id)''')
    result = all_patents[['patent_id']].copy()
    metadata = []
    for count in [1, 2, 3]:
        expression = "concat_ws(' '," + ','.join(
            'words[i' + ('+' + str(index) if index else '') + ']' for index in range(count)) + ')'
        connection.execute('DROP TABLE IF EXISTS grams')
        connection.execute('DROP TABLE IF EXISTS first_seen')
        connection.execute(f'''CREATE TEMP TABLE grams AS SELECT DISTINCT patent_id,grant_date,{expression} term
            FROM tokens CROSS JOIN UNNEST(range(1,len(words)-{count}+2)) p(i)''')
        connection.execute('CREATE TEMP TABLE first_seen AS SELECT term,min(grant_date) first_date FROM grams GROUP BY term')
        features = connection.execute(f'''SELECT g.patent_id,count(*) lexical_distinct_{count},
            sum((g.grant_date=f.first_date)::INT) lexical_new_{count},
            avg((g.grant_date=f.first_date)::INT) lexical_new_share_{count}
            FROM grams g JOIN first_seen f USING(term) GROUP BY g.patent_id''').df()
        result = result.merge(features, on='patent_id', how='left', validate='one_to_one')
        pairs, terms = connection.execute('SELECT (SELECT count(*) FROM grams),(SELECT count(*) FROM first_seen)').fetchone()
        metadata.append({'ngram': count, 'patent_term_pairs': pairs, 'distinct_terms': terms})
    result.to_parquet(output('data/local_lexical_features.parquet'), index=False)
    save('results/lexical_definition.json', {'counts': metadata,
        'definition': 'Distinct term share not seen at a strictly earlier grant date in the available A61B corpus; same-day first users tied.',
        'scope': 'Local A61B title, abstract and first independent claim; not an exact replication of Arts et al.',
        'preprocessing': 'Lowercase alphabetic tokens, minimum length three, scikit-learn English stopwords; no stemming.'})
    connection.close()


if __name__ == '__main__':
    main()
