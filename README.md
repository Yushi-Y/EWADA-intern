# Privacy-preserving Recommendations on Solid 
Develop collaborative filtering (CF) recommendation algorithms to be employed on Solidflix, a Solid-based movie recommendation sharing app. Solid (Social Linked Data) is a decentralised personal data architecture. The aim of the project is to preserve user privacy while using CF.


Directory structure:

- KNN+SVD - Implemented KNN for user-based CF and SVD for matrix factorisation.

- Minhash+LSH - A hash-based algorithm for searching for nearest neighbours in a potentially privacy-preserving manner. 

- User_clustering - Created an algorithm to cluster users based on movie content features, such as popularity, genre, etc.

- Neural_CF - Implemented neural matrix factorisation, and experimented with a simple neural CF model to be run with MP-SPDZ (https://github.com/data61/MP-SPDZ) for multi-party computations. 
