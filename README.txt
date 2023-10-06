1. Python code
- Data_processing1
To process description data, including tokenization, lemmatization, and stopwords removement.
We calculated frequency of words and only selected words appearing over 150 times (about 95 words).
Onehotencoding was conducted on these 95 words.

- Data_processing2
Trying to analyze popularity of movie with different age certifications.
We transformed popularity values by log2.
We generated a boxplot to show the popularity value distribution for different age certification.

- Data_processing3
Comparison of popularity of three categories age certification movies with different genres
We implemented one-way ANOVA and post-hoc test.