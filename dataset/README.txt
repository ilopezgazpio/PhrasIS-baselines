This folder contains the accompanying files for the paper: "PhrasIS: phrase inference and similarity benchmark"


Dataset
-------
10214 phrase pairs with a relation label and a similarity score (20428 labeled samples)

The dataset is divided on a train (6491 pairs) and a test split (3723 pairs).
Each split contains positive phrase pairs (2131 and 1190 respectively) and negative phrase pairs (4360 and 2533 respectively)


Source
------
The phrases have been extracted from the iSTS16 dataset sentence pairs (Agirre et. al, 2016), 
including unaligned pairs (negative pairs) and aligned pairs (positive pairs).
We reannotated the positive pairs out of context, with new score and inference relations.
See paper for details.

The sentence pairs pertain to these two datasets:

- headlines: Headlines mined from several news sources by European Media Monitor using their RSS feed.
  http://emm.newsexplorer.eu/NewsExplorer/home/en/latest.html

- images: The Image Descriptions data set is a subset of the Flickr dataset presented in (Rashtchian et al., 2010),
  which consisted on 8108 hand-selected images from Flickr, depicting actions and events of people or animals,
  with five captions per image.
  The image captions of the data set are released under a CreativeCommons Attribution-ShareAlike license.


Train files
-----------
PhrasIS.train.headlines.negatives.txt
PhrasIS.train.headlines.positives.txt
PhrasIS.train.images.negatives.txt
PhrasIS.train.images.positives.txt

Test files
----------
PhrasIS.test.headlines.negatives.txt
PhrasIS.test.headlines.positives.txt
PhrasIS.test.images.negatives.txt
PhrasIS.test.images.positives.txt

Format
------

The dataset comes in form of a tab separated plain text file.

- similarity score: from 0 to 5
- relation label: one of the following labels {EQUI, FORW, BACK, SIMI, REL, OPPO, UNR}
- prhase1: tokens of first phrase
- prhase2: tokens of second phrase
- token id numbers sequence of first phrase: position id on the first
- sentence of the original sentence pair

token id numbers sequence of second phrase: position id on the second sentence of the original sentence pair
pair number: original sentence pair number in the iSTS16 corpus


References
----------

Eneko Agirre, Aitor Gonzalez-Agirre, Iñigo Lopez-Gazpio, Montse
   Maritxalar, German Rigau, and Larraitz Uria. 2016.  Semeval-2016
   task 2: Interpretable semantic textual similarity. In Proceedings
   of the 10th International Workshop on Semantic Evaluation (SemEval
   2016), San Diego, California, June.

Rashtchian, C., Young, P., Hodosh, M., Hockenmaier, J.,
   2010. Collecting image annotations using Amazon’s Mechanical
   Turk. In: Proceedings of the NAACL HLT 2010 Workshop on Creating
   Speech and Language Data with Amazon’s 39Mechanical Turk. CSLDAMT
   2010. Stroudsburg, PA, USA, pp. 139–147.

