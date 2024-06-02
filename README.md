# General
We use text analytics approach to have a better understanding of GitHub repositories content. We focus on particular repositories of interest, however, the conceptual model is transferrable to any other project domain. We use Python and Python libraries for our computations.

**Motivation**: Motivated by the presence of repositories with different type and scope of content within one GitHub project, e.g., SDK tools  vs documentation repos within Ethereum GitHub account, we aim to develop a classification tool to group repositories and analyze them. Due to the conceptual and technological differences between open-source software projects, there is no unified approach for classification, and existing classification within one project (e.g., Ethereum software development can be classified according to 6 EIPs categories) is not applicable to other projects. Therefore, the requirement is to use unsupervised or self-supervised approach to account for the variety between projects. To achieve this, we employed natural language processing (NLP) techniques on repository READMEs and their descriptions, as well as topic modeling and text clustering applications within text mining.

## Approach 1: State-of the art NLP models
We begin with using machine learning models for topic modeling and text clustering within Ethereum GitHub organization. The conceptual approach is then repeated for other organizations, which demonstrates its generalizability. 

## Step 1. Data collection 
First, we collect repo data from GitHub - repository READMEs and descriptions. We use endpoints 'description' and 'readme' within endpoint 'name' from GitHub API. The sample code is accessible here: [crawler](https://www.dropbox.com/scl/fi/oenp67cz3b2bn8nnb9rse/crawler.py?rlkey=m9vht3padb8u84851vc2tood8&dl=0). Next, we clean the data from some information that may create noise, e.g., calendar dates, web links, etc. (codes:[cleaned](https://www.dropbox.com/scl/fi/ff0ow3vfieetqd418q8uz/clean.py?rlkey=byjjk3jb1hgbdffku5n6ir8ws&dl=0)). We store the clean data as presented in our template file [ethereum_repos_extracted_text_cleaned.json](https://www.dropbox.com/scl/fi/moq9cd1ppfhl66bo3d99g/ethereum_repos.json?rlkey=uf38jgmtjiivqzo8hi47isznu&dl=0).  

## Step 2-1. Repo Clustering
Second, we feed the repos name, description and readme data in the BERT model. BERT is a popular word embedding machine learning model. After, the text data is processed, we want to classify the most similar repositories into groups. To do that, we use K-means clustering approach, where number of clusters K is identified by the "elbow" method and KneeLocator function. We also used dendrogram to present agglomerative clustering. We first used both readme and description of each repo as analyzing texts [unsupervised (desc and readme)](https://www.dropbox.com/scl/fi/trwoo89mh8hc2hw52lt7h/unsupervised_-only_desc.ipynb?rlkey=xvooyqd4lwyrmm5rykij5p226&dl=0). Here is the cluster results in csv format, displaying cluster number and the repos underneath: [clustering results](https://www.dropbox.com/scl/fi/6b9lignrqih7k6zd06lqa/Clustering-Results-unsupervised-using-t-SNE-csv.csv?rlkey=cxakx8osdnkjaa0pvfxd0682d&dl=0).

Nevertheless, feeding the whole text of the repo readme leads to overfitting. To conquer this problem, we feed only desciption in the BERT model [unsupervised(only desc)](https://www.dropbox.com/scl/fi/trwoo89mh8hc2hw52lt7h/unsupervised_-only_desc.ipynb?rlkey=xvooyqd4lwyrmm5rykij5p226&dl=0). The file with descriptions used in code is here: [ethereum_repos_extracted_text_cleaned.json](https://www.dropbox.com/scl/fi/moq9cd1ppfhl66bo3d99g/ethereum_repos.json?rlkey=uf38jgmtjiivqzo8hi47isznu&dl=0). 

We plot the clusters of repos with their name displayed as shown in figure attached here: [text_clustering](https://www.dropbox.com/scl/fi/vtj90cjawe16gekaxcaqn/textdata-1-1.pdf?rlkey=y7arov7mqmqp3joytaznnycmt&dl=0). The dendrogram figure is presented here as well: [dendrogram](https://www.dropbox.com/scl/fi/p5l5zmvlrxrwu535773vf/Dendrogram-1.pdf?rlkey=e21aklbp4s0cbpal2ucrs7yw3&dl=0)

The repo clustering approach yields repo clusters, however, we still do not know the common themes within created clusters. Next, we discuss topic modeling approach.

## Step 2-2. Topic Modeling
Next, we use latent Dirichlet allocation (LDA) model to come up with topics for clusters. LDA approach alloows to both cluster repositories based on their textual description and extract common topics (keywords) that characterize each cluster. We compute the coherence score to select the optimal number of clusters (analogy to "elbow" rule). Among 2-10 topics, we found that classification into 7 clusters has the highest coherence score. Also, it has a low perplexity score. The codes can be found in [Topic Modeling](https://www.dropbox.com/scl/fi/d1852g78rbtcggky5u27h/Topic_modeling.ipynb?rlkey=aas4t1mruf0vu7fsl3eeyl6tj&dl=0).

We aimed to enhance the unsupervised classification approach, that is why we trained a Doc2Vec model using the Gensim library on ERC and EIP documents to create document embeddings that serve our classification objectives. Here is the code:[Doc2Vec](https://www.dropbox.com/scl/fi/1q1s4z7ys9ldwo6m63bvv/doc2vec.py?rlkey=z0yu5z2deeu9s6szh4k2c6bjd&dl=0). Here is the [folder](https://www.dropbox.com/scl/fo/h5mthfaptplxdnk2ebjwm/AJ5go2TyT2YeOo7fza_FekE?rlkey=ruyvrc9lbkpj4ti0e71ia16qu&dl=0) containing ERC and EIP documents to run the codes. 

## Step 3. Replication
Similar steps can be performed with any GitHub organization. To receive common classification among several GitHub organizations, a similar algorithm of repo cluster classification and topic modeling can be applied to created descriptions of clusters within organizations.

## Approach 2: Transformer model
After seeing the results of the previous approach, we develop alternative text classification approach using generative pre-trained trasformer (GPT) model form OpenAI, i.e., GPT-3/3.5/4. The benefit of the model is ease of use and the superior text processing ability. On the other hand, the probabilistic nature of generative model may create additional challenges that we discuss at the end.

## Step 1: ChatGPT Approach to Classifying Repositories: From Named Clusters to No Prior Labels
After collecting data as indicated in the step 1 to previous approach, we connect to the ChatGPT API (https://openai.com/blog/introducing-chatgpt-and-whisper-apis). The GPT approach requires us to create textual promts as instructions for the model. 
We develop three scenarios with various level of supervision on the number of clusters. The code for this step is accessible here: [Chatgpt Approach](https://www.dropbox.com/scl/fi/zxzldsc6qrl70y43r5py1/chatgpt_approach.ipynb?rlkey=iscif9x5fojk857tnup7wvxi6&dl=0)

**Scenario 1**: 
In scenario 1, we start with "supervised" prompt by explicitly identifying cluster topics based on Ethereum EIP categories (https://eips.ethereum.org): Core, Network, Interface, ERC, Meta, and Informational. This method is the most predictable regarding our anticipated clustering of categories. Yet,its applicability to other open-source software projects is limited without predefined categories from external sources. To address this constraint, we have developed additional scenarios (scenario2 and scenario3).
We upload "ethereum_repos.json" for running this code block, which is a file consisting of repo name, description and readme. We loop through all repos and ask chatgpt to classify each repo to its corresponding cluster. The result is presented in file: [senario1.json](https://www.dropbox.com/scl/fi/rv4pnff9tddubqehi4bko/senario1.json?rlkey=wlr8t9ubk9defz27j7s1u3j1b&dl=0). And here is the reorganized csv version:[Scenario1.csv](https://www.dropbox.com/scl/fi/h7w2bcvp1fekv4t0rfudy/Scenario1.csv?rlkey=76vftz4mdlhuccx3xlwnc3qbz&dl=0).

**Scenario 2**:
In scenario 2, we continue with a "semi-supervised"" prompt, where we ease the
constraints on the number of categories and their respective topics. While we continue to
recommend the initial six categories to the model, we now permit the model to generate
additional clusters if there are repositories that support the formation of these new groups.
Due to token limits, we only upload repo description to feed into ChatGPT. Based on
all these repo descriptions, chatgpt will come up with several cluster topics.
As a result, ChatGPT model did not give us additional clusters besides the 6 clusters we proposed to it. 

**Scenario 3**:
In scenario 3, we relax all the constraints in the prompt allowing the model decide itself, in other words, highly autonomous,
minimally supervised approach. In this scenario, we permit the model to classify without
predefined categories, maintaining an upper limit on the number of categories to ensure
they remain interpretable by humans.
As a result, ChatGPT creates 15 clusters. Here is the initial results displayed: [Scenario 3 initial result](https://www.dropbox.com/scl/fi/0ug3nout96pzijgm6fssi/Scenario-3-initial-results.txt?rlkey=95wq3mx654r7effnfzvcv3deo&dl=0We). Then we loop through each repo and get them classified. Here is the result grouped by cluster name: [Scenario 3  output csv](https://www.dropbox.com/scl/fi/0ug3nout96pzijgm6fssi/Scenario-3-initial-results.txt?rlkey=95wq3mx654r7effnfzvcv3deo&dl=0)

## Step 2: Expand to More Organizations 
After we have some results for one organization, we apply the algorithm above to a sample of other organizations on GitHub.

**Data collection** We follow the approach described above and collect information for a sample of OSS accounts to measure their similarity, including their **repo name**, **description**, **readme**, **license info**(including license_key, license_name, licsense_spdx id, license_url, license_node id), **language** and **release dates**. Here is the folder holding the codes and results for crawling: [crawling for 31 orgs code and results in json format](https://www.dropbox.com/scl/fo/htajucd61xfp6opv5hi6o/AJ6Wy7fYrdmGtdIlb8pUZKo?rlkey=8h3gnz9t8nux688imwpsi0pms&dl=0). To have a better view for the info, we convert this info to csv for each org. In each csv, each row represents a repo belonging to this org, and the columns contain their info listed before as bold. Here is the code link. Here are the codes for conversion and the final output in csv format: [orgs info csv](https://www.dropbox.com/scl/fo/iu7rvkfb5qgkug7vr1qr7/AJJbXUBqvSoFWNaI8uFeS-k?rlkey=ofdj0afdu1k0ubrpvmbwfe440&st=gfl81qyo&dl=0)

**Classification within organizations** We begin with a small sample of 7 chosen organizations to come up with clusters for repositories within each of them separately. The sample includes Algorand, cardano-foundation, dashpay, zcash, stellar, tezos, and solana-labs. Following the approach described above, we use ChatGPT to come up with clusters for each orgnaization. Our goal is to receive the interpretable clusters of repositories, so we provide some guidance to the ChatGPT by setting up the upper boundary for the number of clusters that would be easily interpreted by the human. For each organization, we receive up to 6 clusters and the list of repositories within the organization for each cluster. We store the classification results in a CSV file [other orgs clustering trials](https://www.dropbox.com/scl/fo/mfvc63dmg4a2zmln7x6hn/AIZFj2l5495cHx_yJWGqJnU?rlkey=3hd6ki6ty2z3nar1zmxykmkw0&dl=0). Overall, the approach seems technically reasonable and produces reasonable clusters like Development Environment. We further scale this approach.

**Classification across organizations** We extend the sample to 30 OSS organizations with total number of 9762 repositories. In this iteration, we pool together the repositories across all organizations and apply the classification algorithm using the repositories description. We aim to show the generalization of the clustering approach across any randomly selected sample of repositories. Nevertheless, due to the heavy labeling of the repositories description, i.e., use of the organization names across repositories within the same organization, this may add additional noise to the data and lead to trivial classification based on organizations belonging. We ran the algorithm to check this hypothesis.
 
**Classification results** Due to the large number of repositories, the dataset of 9762 of repositories exceeded the token limits of both ChatGPT 3.5 and 4.0, resulting in an error message indicating that the request was too large(when we tried to use compression, the limit of chatgpt 4 is 10000 while our input is 173373). Consequently, we sampled 500 repositories across all 30 organization proportional to the size of the organization. These weights were determined by inversely weighting them by the number of repositories per organization. The sample code is presented here: [repo_reclassify_sampled](https://www.dropbox.com/scl/fi/js5e0m3bkgr9ysmkpp8om/repo-_reclassify_sampled.py?rlkey=rv2suo96eum0sbabzbo5l6qcb&dl=0). As our results show, our ChatGPT approach creates four clusters: Cluster 1: Software Development and Programming Languages, Cluster 2: Documentation, Information and Learning Resources,Cluster 3: Machine Learning & AI, Cluster 4: Blockchain and Cryptocurrency technologies. These clusters are not based on organizations but more of the primary technological domain or application area that each repository contributes to. the classification results are presented here：[parent_project](https://www.dropbox.com/scl/fi/qrrpp2a2vz7j9cdc6ovcp/parent_project.xlsx?rlkey=nbjaxmn0h95k3aq7eg8n1qshy&dl=0). (potential insights include cluster distribution in orgs)

**text clustering**: We also tried text clustering approach to reclassify the sampled 500 repos as discussed right above. We used repos descriptions and used the Within-Cluster Sum of Squares to decide the optimal clusters. There are 7 clusters, some focus on various development tools and technologies, some on Linux and Coding Infrastructure, and some on Documentation and APIs,some on Machine Learning and Experimental Hypotheses, some on Cryptography and Security, etc. When we used use the BERT model to obtain contextual representations of words and the optimal number for clusters in this case is 6. Then, we also use t-SNE for dimensionality reduction and visualized the clusters. The code and results are here:[repo_reclassify_sampled_unsupervised](https://www.dropbox.com/scl/fi/z7npr9shdlar7z5lnbbhj/repo_reclassify_sampled_unsupervised.ipynb?rlkey=5gfnu4z0slj3v7mrvom1jsceh&dl=0).

Both the ChatGPT approach and text clutsering show the classification from this randomly chosen sample gives us broader and generalized clusters, which are not highly based on repos' parent organizations' name.
 

**pinner repos trials**: To address the repositories selection bias, we try other sampling approaches. Also, we want to avoid the bias of choosing specific orgs as opposed to more repos. Then, we tried another way, which is selecting all pinned repos for the 31 orgs as information feed. We retrieved the pinned repo from github and feed their info to Chatgpt model. We have several experiments in terms of information feed (
with description, name, with or without orgs name, with or without repo name), (with or without a rough instruction to tell it: do not only rely on original orgs name). What can be seen clearly to us is that, if we feed into orgs names, chatgpt mostly likely gives the clusters only based on original orgs. It seems like the first trial (feed in repo name and description, without orgs name) results in some broader definition for clusters, but compared to the randomly sampling 500 repos case, it has much more focus on the original orgs. Here are the codes: [pinned_repos](https://www.dropbox.com/scl/fi/q1mz5sipm8l9aohhe0v0k/pinned_repos.ipynb?rlkey=qq82audcsqskrgttaxm5iakmj&dl=0)

This trial and result discussed above further support our hypothesis: Due to the use of organization names in the descriptions (not to mention adding organizations' name as information feed) of multiple repositories within the same organization, there may be added noise in the clustering process, potentially resulting in superficial classifications based on organizational affiliation.


**To be continued:** feed all 9k to see how the clusters would be. We might want to consider feeding in repos in batch. 

**side note:** we also retrieved comprehensive dataset similar to what we have for Ethereum and Bitcoin analysis for other orgs, which can be seen [other_org_all_data](https://www.dropbox.com/scl/fo/1ovgi8iges897vdrtn0cg/AHlhOZrHcjWqoudc-xCO0vQ?rlkey=93va7357gazcia8mn1o5f2l5f&st=ksuex373&dl=0) from public dropbox










