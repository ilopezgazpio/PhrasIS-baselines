''' Constants for column groups '''

LEXICAL_COLS = ['jaccard_strip_tokenized', 'jaccard_strip_tokenized_noPunct_lemmat_noStopWords', 'jacckard_strip_tokenized_noPunct']
WORDNET_PATH_COLS = ['path_similarity', 'path_similarity_root']
WORDNET_LCH_COLS = ['lch_similarity_nouns', 'lch_similarity_verbs', 'lch_similarity_nouns_root', 'lch_similarity_verbs_root']
WORDNET_JCN_COLS = ['jcn_similarity_brown_nouns', 'jcn_similarity_brown_verbs', 'jcn_similarity_genesis_nouns', 'jcn_similarity_genesis_verbs']
WORDNET_WUP_COLS = ['wup_similarity', 'wup_similarity_root']
WORDNET_DEPTH_COLS = ['chunk1>chunk2', 'chunk2>chunk1', 'minimum_difference', 'maximum_difference']
LENGTH_COLS = ['left-right', 'right-left', '|chunk1-chunk2|']

ALL_FEATURES = LEXICAL_COLS + WORDNET_PATH_COLS + WORDNET_LCH_COLS + WORDNET_JCN_COLS + WORDNET_WUP_COLS + WORDNET_DEPTH_COLS + LENGTH_COLS