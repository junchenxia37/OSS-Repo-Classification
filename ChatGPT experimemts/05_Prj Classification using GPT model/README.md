# Start with Ethereum 
used text analytics to have a better understanding of potential repos classification under this specific organization.

**Motivation**: Although there is ongoing discussions on how to categorize Ethereum Improvement Proposals (EIPs), there is no established consensus on the best method for classifying repositories within the Ethereum organization. Rather than simply adopting the EIP classifications, we aimed to develop a more technical approach to classification beyond heuristic methods. To achieve this, we employed unsupervised learning: topic modeling and text clustering techniques on repository READMEs and descriptions.

## Topic Modeling
 We utilized topic modeling to derive several topics from ethereum repos. The codes can be found in [Topic Modeling](https://www.dropbox.com/scl/fi/d1852g78rbtcggky5u27h/Topic_modeling.ipynb?rlkey=aas4t1mruf0vu7fsl3eeyl6tj&dl=0). To run this code, we need to import what we have extracted from GitHub Ethereum (codes for crawling is [crawler](https://www.dropbox.com/scl/fi/oenp67cz3b2bn8nnb9rse/crawler.py?rlkey=m9vht3padb8u84851vc2tood8&dl=0)), and we cleaned it (codes:[cleaned](https://www.dropbox.com/scl/fi/ff0ow3vfieetqd418q8uz/clean.py?rlkey=byjjk3jb1hgbdffku5n6ir8ws&dl=0)), the results are in the file named [ethereum_repos_extracted_text_cleaned.json](https://www.dropbox.com/scl/fi/moq9cd1ppfhl66bo3d99g/ethereum_repos.json?rlkey=uf38jgmtjiivqzo8hi47isznu&dl=0). We used LDA model to come up with several topics within ethereum repos. How to determine the optimal topics number? We compute the coherence score. Among 2-10 topics, we found 7 has the highest coherence score. Also, it has a low perplexity score. 


 ## Repo Description and Readme Clustering 
 We used BERT word embedding to process repos name, description and readme. Then we used k-means for clustering, we used elbow method and KneeLocator to find the optimal cluster numbers. We also used dendrogram to present agglomerative clustering. We first used both readme and description of each repo as analyzing texts [unsupervised (desc and readme)](https://www.dropbox.com/scl/fi/trwoo89mh8hc2hw52lt7h/unsupervised_-only_desc.ipynb?rlkey=xvooyqd4lwyrmm5rykij5p226&dl=0). Here is the cluster results in csv format, displaying cluster number and the repos underneath: [clustering results](https://www.dropbox.com/scl/fi/6b9lignrqih7k6zd06lqa/Clustering-Results-unsupervised-using-t-SNE-csv.csv?rlkey=cxakx8osdnkjaa0pvfxd0682d&dl=0).
 
 Then, since feeding all repos readme leads to overfit sometimes, we wanna try a more precise version.That is why we used only desciption [unsupervised(only desc)](https://www.dropbox.com/scl/fi/trwoo89mh8hc2hw52lt7h/unsupervised_-only_desc.ipynb?rlkey=xvooyqd4lwyrmm5rykij5p226&dl=0). Though the two codes only have slight different, we uploaded the ipynb to show the results. To run these two code, we need to import what we have extracted, which is the file named [ethereum_repos_extracted_text_cleaned.json](https://www.dropbox.com/scl/fi/moq9cd1ppfhl66bo3d99g/ethereum_repos.json?rlkey=uf38jgmtjiivqzo8hi47isznu&dl=0). The visulization of repos clustering with their name displayed can be found in this SVG [text_clustering](https://www.dropbox.com/scl/fi/vtj90cjawe16gekaxcaqn/textdata-1-1.pdf?rlkey=y7arov7mqmqp3joytaznnycmt&dl=0).The dendrogram is here [dendrogram](https://www.dropbox.com/scl/fi/p5l5zmvlrxrwu535773vf/Dendrogram-1.pdf?rlkey=e21aklbp4s0cbpal2ucrs7yw3&dl=0)




 We aimed to enhance the unsupervised classification approach, that is why we trained a Doc2Vec model using the Gensim library on ERC and EIP documents to create document embeddings that serve our classification objectives. Here is the code:[Doc2Vec](https://www.dropbox.com/scl/fi/1q1s4z7ys9ldwo6m63bvv/doc2vec.py?rlkey=z0yu5z2deeu9s6szh4k2c6bjd&dl=0). Here is the [folder](https://www.dropbox.com/scl/fo/h5mthfaptplxdnk2ebjwm/AJ5go2TyT2YeOo7fza_FekE?rlkey=ruyvrc9lbkpj4ti0e71ia16qu&dl=0) containing ERC and EIP documents to run the codes. 


## ChatGPT Approach to Classifying Repositories: From Named Clusters to No Prior Labels
We want to use the most powerful tool "chatgpt" to help us classify repos.
Using ChatGPT API (https://openai.com/blog/introducing-chatgpt-and-whisper-apis) and GitHub API GitHub API(https://docs.github.com/en/rest), we develop a Python program that retrieves the readme file content for each repository and feeds it into the algorithm below.
There are 3 scenarios from supervised to unsupervised approach. Here are the codes: [Chatgpt Approach](https://www.dropbox.com/scl/fi/zxzldsc6qrl70y43r5py1/chatgpt_approach.ipynb?rlkey=iscif9x5fojk857tnup7wvxi6&dl=0)

**Scenario 1**: 
First, we start with supervised approach by giving cluster topics based on
Ethereum EIP categories (https://eips.ethereum.org): Core, Network, Interface, ERC, Meta,
and Informational. This method is the most predictable regarding our anticipated clustering of categories. Yet,its applicability to other open-source software projects is limited without predefined categories from external sources. To address this constraint, we have developed additional scenarios (scenario2 and scenario3).
* Note: we upload "ethereum_repos.json" for running this code block, which is a file
consisting of repo name, description and readme

We loop through all repos and ask chatgpt to classify each repo to its corresponding cluster.
The result is [senario1.json](https://www.dropbox.com/scl/fi/rv4pnff9tddubqehi4bko/senario1.json?rlkey=wlr8t9ubk9defz27j7s1u3j1b&dl=0). And here is the reorganized csv version:[Scenario1.csv](https://www.dropbox.com/scl/fi/h7w2bcvp1fekv4t0rfudy/Scenario1.csv?rlkey=76vftz4mdlhuccx3xlwnc3qbz&dl=0).


**Scenario 2**:
We continue with a semi-supervised approach. In scenario 2, we ease the
constraints on the number of categories and their respective topics. While we continue to
recommend the initial six categories to the model, we now permit the model to generate
additional clusters if there are repositories that support the formation of these new groups.
* Note: due to token limits, we only upload repo description to feed into chatgpt. Based on
all these repo descriptions, chatgpt will come up with several cluster topics.

Chatgpt model did not give us additional clusters besides the 6 clusters we proposed to it. 

**Scenario 3**:
we explore an unsupervised approach, in other words, highly autonomous,
minimally supervised approach. In this scenario, we permit the model to classify without
predefined categories, maintaining an upper limit on the number of categories to ensure
they remain interpretable by humans.

In terms of results for scenario 3. Chatgpt gives 15 clusters. Here is the initial results displayed: [Scenario 3 initial result](https://www.dropbox.com/scl/fi/0ug3nout96pzijgm6fssi/Scenario-3-initial-results.txt?rlkey=95wq3mx654r7effnfzvcv3deo&dl=0We). Then we loop through each repo and get them classified. Here is the result grouped by cluster name: [Scenario 3  output csv](https://www.dropbox.com/scl/fi/0ug3nout96pzijgm6fssi/Scenario-3-initial-results.txt?rlkey=95wq3mx654r7effnfzvcv3deo&dl=0)

# Expand to More Organizations 

After we have some results for Ethereum, we wanted to apply these techniques to other orgs to scale up.

**Gather info for 30 orgs:** We wanted to see the similarity and differences among different open source software organizations. We first crawling the info for 30 organizations (there are 31 orgs in the initial orgs list, however, we decided to skip Apache since it is too large to process at first), including their **repo name**, **description**, **readme**, **license info**(including license_key, license_name, licsense_spdx id, license_url, license_node id), **language** and **release dates**. Here is the codes and results for crawling: [crawling for 31 orgs code and results in json format](https://www.dropbox.com/scl/fo/htajucd61xfp6opv5hi6o/AJ6Wy7fYrdmGtdIlb8pUZKo?rlkey=8h3gnz9t8nux688imwpsi0pms&dl=0). To have a better view for the info, we convert this info to csv for each org. In each csv, each row represents a repo belonging to this org, and the columns contain their info listed before as bold. Here is the code link. Here are the codes for conversion and the final output in csv format: [orgs info csv](https://www.dropbox.com/scl/fo/ozuec0ahgx3iku91narfi/ABmRed5KudSnaSTJ5ugYMiU?rlkey=91rzh8zixpf5ky3sfad996y8x&dl=0)

**7 orgs for initial trial:** After we have the relevant info for these orgs. We wanted to come up with some common clusters across the orgs. We chose 7 orgs for experiments due to familarity to get a sense of how well the clusters work. They are Algorand, cardano-foundation, dashpay, zcash, stellar, tezos, and solana-labs. We used Chatgpt model to come up with up to 6 clusters for each orgs. We wanted to find some similar clusters among those orgs. So for each orgs, we have its up to 6 clusters with repos under that cluster. We made a CSV file to make the results clear and ready to sort to compare the similarity of clusters between orgs. We recorded the codes and results for the "try 7 orgs" in Folder [other orgs clustering trials](https://www.dropbox.com/scl/fo/mfvc63dmg4a2zmln7x6hn/AIZFj2l5495cHx_yJWGqJnU?rlkey=3hd6ki6ty2z3nar1zmxykmkw0&dl=0). Taking a glance at these results, it worked well. There are some common clusters like Development Environment. More insights to be concluded.

This inspired us to explore more clustering stuff across more orgs. Right now, we have 31 orgs with repos belong to them natually. **Our hypothesis is that the cluster will be 31 as repos will be clustered just to its corresponding org**. 

As previous experiments suggests, we will use repos' decription as information feed instead of readme as readme takes too many tokens space. 

Our initial intention was to feed all repository descriptions from 31 organizations (except Apache, Ethereum repos included, more than 9k repos in total), to ChatGPT in order to generate multiple clusters. However, this task proved to be beyond the capacity limits of both ChatGPT 3.5 and 4.0, resulting in an error message indicating that the request was too large. 

 
**repo reclassify sampled:** Consequently, we sampled 500 repositories (as 800 repos exceeded the limits too). The 500 repositories were selected using weights to guarantee a balanced representation across organizations of different sizes. These weights were determined by inversely weighting them by the number of repositories per organization, aiming to ensure fairness in the selection process. (That is our adpoted approach, however, if the perspective is to amplify the voice of larger organizations, default weights work). Here is the code: [repo_reclassify_sampled](https://www.dropbox.com/scl/fi/js5e0m3bkgr9ysmkpp8om/repo-_reclassify_sampled.py?rlkey=rv2suo96eum0sbabzbo5l6qcb&dl=0).  We got the results from this sampling, we tried to set seed, but it did not work well, to be explored. In one trial we recorded, Chatgpt model comes up with 4 clusterings: Cluster 1: Software Development and Programming Languages, Cluster 2: Documentation, Information and Learning Resources,Cluster 3: Machine Learning & AI, Cluster 4: Blockchain and Cryptocurrency technologies. These clusterings are not based on organizations but more of the primary technological domain or application area that each repository contributes to. 


Then, we map the repos displayed under the clusters Chatpgt model come up with to their corresponding parent project, and notify its new cluster, in other words, the new cluster it belongs to. We created the excel file here for a review：[parent_project](https://www.dropbox.com/scl/fi/qrrpp2a2vz7j9cdc6ovcp/parent_project.xlsx?rlkey=nbjaxmn0h95k3aq7eg8n1qshy&dl=0). (potential insights include cluster distribution in orgs)

**pinned_repos trials**: Also, we recognized the limitations of the randomness of the selected 500 repos and its representativeness. Also, we want to avoid the bias of choosing specific orgs as opposed to more repos. Then, we tried another way, which is selecting all pinned repos for the 31 orgs as information feed. We retrieved the pinned repo from github and feed their info to Chatgpt model. We have several experiments in terms of information feed (
with description, name, with or without orgs name, with or without repo name), (with or without a rough instruction to tell it: do not only rely on original orgs name). What can be seen clearly to us is that, if we feed into orgs names, chatgpt mostly likely gives the clusters only based on original orgs. It seems like the first trial (feed in repo name and description, without orgs name) results in some broader definition for clusters, but compared to the randomly sampling 500 repos case, it has much more focus on the original orgs. Here are the codes: [pinned_repos](https://www.dropbox.com/scl/fi/q1mz5sipm8l9aohhe0v0k/pinned_repos.ipynb?rlkey=qq82audcsqskrgttaxm5iakmj&dl=0)
 
**text clustering**: We also tried text clustering approach to reclassify the 500 repos as discussed right above. We used repos descriptions and used the Within-Cluster Sum of Squares to decide the optimal clusters. There are 7 clusters, some focus on various development tools and technologies, some on Linux and Coding Infrastructure, and some on Documentation and APIs,some on Machine Learning and Experimental Hypotheses, some on Cryptography and Security, etc. When we used use the BERT model to obtain contextual representations of words and the optimal number for clusters in this case is 6. Then, we also use t-SNE for dimensionality reduction and visualized the clusters. The code and results are here:[repo_reclassify_sampled_unsupervised](https://www.dropbox.com/scl/fi/z7npr9shdlar7z5lnbbhj/repo_reclassify_sampled_unsupervised.ipynb?rlkey=5gfnu4z0slj3v7mrvom1jsceh&dl=0).

**To be continued:** feed all 9k to see how the clusters would be










