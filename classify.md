Classifying Hacker News Titles
Introduction
In my last essay I described the problem of filtering Hacker News to produce a purified feed. My viewpoint, motivation and idea of what constitutes a "good" posting is described there, so read it if you're interested.
Since then I've developed a classifier that produces an enriched feed. In this article I'll describe the classifier, show you the results, and show some evaluation information.
The overall methodology is like that used in the machine learning literature, where I split a labeled sample of articles into a training and a test set, create a model, and then test the model by comparing predictions it makes to my own evaluations.

what's unusual about this article is that it applies this technique to a problem that I formulated from the beginning, applying the evaluation methodology to confirm fitness to purpose, as opposed to entering into a competition to get better results on a pre-existing data set.
Along the way I will explain how my model works, what the output that I see looks like, and how some changes to the system affect the quality of results, with an eye to understanding what the training requirements for a more general feed classifier would be.
A working model
Here is the information that I have for an article in the training set:
stories/15608169
relevant	True
href	http://dark.nightmare.com/rushing/irken/irken/lang.html
title	Irken – A statically-typed Scheme with row polymorphism
hnuser	jokermatt999
words	[irken, –, a, statically-typed, scheme, with, row, polymorphism]
Input Features
My classifier only looks at tokens that appear in the title. The tokens are produced by the nltk_tokenize function: some of these tokens are words, others are punctuation. It is common in this kind of work to eliminate the most common words (stopwords) and to eliminate words that are less common (because these are not well sampled.) For now, I keep everything, because the classifier I am using (logistic regression) can handle a large number of features, both in terms of execution speed and in terms of realistically estimating how much weight to put on both highly common and highly rare features.
For each word in the vocabulary, there is a feature: the value of that feature is 1 if the word is present in the title, and 0 if it is not. The input matrix looks like:
index	the	:	a	to	and	of	for	in	,	–	...	giants	campus	moocs	autumn	jihadist	recursive	zalando	norcal	deliver	apocalypse
story																					
stories/15275572	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15624497	0	1	1	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15359134	1	0	0	0	0	1	0	1	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15454882	1	0	0	0	0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15523617	0	0	1	0	0	0	0	0	1	0	...	0	0	0	0	0	0	0	0	0	0
stories/15436767	0	0	0	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15501411	0	1	0	0	0	0	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15277669	0	0	0	0	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15415733	0	0	0	0	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15284015	0	0	0	0	0	0	0	1	0	0	...	0	0	0	0	0	0	0	0	0	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
stories/15452637	1	0	0	0	0	0	0	0	0	1	...	0	0	0	0	0	0	0	0	0	0
stories/15597265	0	0	0	0	0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15602620	0	0	1	0	0	0	0	0	0	1	...	0	0	0	0	0	0	0	0	0	0
stories/15457218	0	1	0	0	0	0	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15410558	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15411600	0	0	0	0	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15273623	0	0	0	1	0	0	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15603534	1	0	0	0	0	1	0	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15293287	0	0	0	1	0	1	1	0	0	0	...	0	0	0	0	0	0	0	0	0	0
stories/15285767	1	0	0	0	0	0	1	1	0	1	...	0	0	0	0	0	0	0	0	0	0
4616 rows × 11216 columns
Alternative bag-of-words features
I'll quickly review some things that I could be doing differently, but am not.
Many text retrieval systems perform stemming:
A stemmer for English, for example, should identify the string "cats" (and possibly "catlike", "catty" etc.) as based on the root "cat", and "stems", "stemmer", "stemming", "stemmed" as based on "stem". A stemming algorithm reduces the words "fishing", "fished", and "fisher" to the root word, "fish". On the other hand, "argue", "argued", "argues", "arguing", and "argus" reduce to the stem "argu" (illustrating the case where the stem is not itself a word or root) but "argument" and "arguments" reduce to the stem "argument".
-- Wikipedia
Intuitively, stemming should improve the performance of a classifier when the number of training examples is small. For instance, if "cat" is a relevant word for a category, "cats" probably is too, and we'll collect better statistics for word occurence if we group them together. On the other hand, this assumption might not always be true: for instance, it could be that "CAT" bulldozers are what's relevant. Also, most stemmers do some violence to the text by committing some errors.
Another possible improvement would be to ignore certain words. For instance, there is little value in words that occur only once or a few times, since whether they appear or not in documents which are relevant or not is just a matter of chance. Words that occur very often (say "the") are also unlikely to be predictive and are frequently treated as stop words.
Some ML algorithms do poorly when they are confronted by a very large number of features, others do better. Logistic regression, the algorithm I use, holds up pretty well, particularly if it is predictions that we want as opposed to insights.
Sticking with the bag-of-words model, I could also weight words based on how often they occur in the title (So that The Global Fund to Fight AIDS, TB and Malaria: A Response to Global Threats, a Part of a Global Future would count the word "Global" three times.) I could also weight words less if they occur frequently in the corpus as well as normalize the length of the title vector. Put those together and I'd get a tf-idf vector.
The case for the tf-idf vector could be stronger if I was using whole documents, but my experience and reading of literature seems to indicate that these choices don't usually make a big difference in the performance of the classifier.
Beyond bag-of-words
Another place I could be going is beyond bag-of-words, to features that are sensitive to the position and order of words.
One scourge of Hacker News is articles of the form "X is down"; it might make sense to build this in as a handcrafted feature, but asking an algorithm to find it is tricky because the number of possible multiple-word phrases expands exponentially as the number of words increases. The training samples thus get divided more and more finely, and it is hard to tell if correlations we see are real or just the result of luck.
People get small amounts of lift on classification tasks like this using features such as Word2Vec, but I am currently unimpressed. To do significally better than bag-of-words, we need something that models and understands language pretty well.
Output labels
The training labels are 1 if an article is considered relevant,  0 if it is not. Here is a sample of the labels that the classifier is trained and tested on:
relevant	title
story		
stories/15496663	0	Rent is the answer
stories/15608169	1	Irken – A statically-typed Scheme with row pol...
stories/15541887	0	Has the Title Engineer Become Meaningless?
stories/15274926	0	The Asshole’s Guide to Product Validation: How...
stories/15388767	0	VR Is Exactly Where It Needs to Be
...	...	...
stories/15432570	0	Licenses
stories/15352624	0	Taking Privacy Seriously: Leaving Google Behind
stories/15279484	1	Tips for giving your first code reviews
stories/15284129	0	Scientific way to prioritize software developm...
stories/15342649	0	Notes from an Apocalypse
5771 rows × 2 columns
Note that our task is a bit unfair for two reasons:
Its not always possible to tell what an article is about from the title. Some titles are descriptive, others are not.
Sometimes I evaluate articles by looking at just the title, but often I read the article. My judgement on an article is not just based on relevance, but also on quality. There are many spammy websites full of tensorflow tutorials, whistlestop summaries of the data science literature stuffed with ads, etc. I vote them down.
If I were just trying to replicate my ability to look at a title and guess at relevance and I wanted to tune it up to maximum accuracy, I would change the task. I might create a "neutral" category and only judge documents on the basis of the title.
The classifier
Fortunately, the classifier I use is good at learning probabilities. Thus, if some fraction of articles with "Tensorflow" in the title are irrelevant, it builds that into its probability estimate.
The classifier I use is the Logistic Regression classifier from scikit-learn with the default parameters. I've tried other parameters and found settings that perform worse, but none that perform markedly better.
Note the decision function looks like:
β0+β1x1+β2x2+⋯+βmxm=β0+∑i=1mβixi
where the  β (s) are the learned coefficients and the  x (s) are the inputs. Note that the  x (s) are zero or one depending on the absence or presence of a token, so the output of the classifier is the sum of the weights for all of the words that appear in the title added to the intercept  β0  and the document is considered relevant if the sum is greater than zero and irrelevant if it is less.
Note there are other algorithms such as linear regression, linear support vector classifier, multinomial naive bayes, perceptron, et al. that train the same decision function but may choose different weights. These algorithms may generalize better or worse than logistic regression; in the little testing I have done these have not performed notably better in accuracy but have often performed much worse in terms of CPU time.
Negative weights, therefore, indicate words that are found in irrelevant titles, and positive weights indicate words found in relevant titles:
coefficient
index	
go	-1.384093
security	-1.357270
things	-1.088976
learn	-1.074222
world	-1.073974
unity	-1.049862
website	-1.008311
vs.	-0.979719
digital	-0.978315
right	-0.888086
...	...
keras	1.386110
feedback	1.429219
conversational	1.430680
deep	1.449445
json	1.490234
gpus	1.508770
library	1.607861
pytorch	1.608896
python	1.671314
analyzing	1.749520
11216 rows × 1 columns
Adding up these weights produces a log-odds score, which can be converted to a probability with an inverse logit function. If the log-odds function is zero, the predicted probability is 0.5; the naive way to get predictions is to predict that the article is relevant if the log-odds is positive, and negative if it is zero.
prediction	log-odds	probability	title
story				
stories/15574717	1.0	0.257790	0.564093	Show HN: Hack to add search to remarkjs slides
stories/15286237	0.0	-2.728917	0.061288	Top Rated Code Editors for Developers
stories/15560193	0.0	-5.000396	0.006690	Are you really middle class? 1960s standard of living now affordable to very few
stories/15471121	0.0	-3.242301	0.037605	The Dystopia Next Door
stories/15521303	0.0	-2.163712	0.103057	Elementary OS My Dear Watson
stories/15468297	0.0	-0.297099	0.426267	The tf.data API [slides]
stories/15488340	0.0	-1.705732	0.153718	Terraform Elasticsearch AWS Module Released
stories/15335048	0.0	-3.425640	0.031504	​ATO resolves 'intermittent issues' affecting its online services, again
stories/15298208	0.0	-2.162536	0.103166	Event Sourcing for the Internet of Things
stories/15331005	0.0	-1.690279	0.155739	Using Big Data to Solve Economic and Social Problems
...	...	...	...	...
stories/15635102	0.0	-1.019417	0.265141	PoC||GTFO 0x16 [pdf]
stories/15316694	0.0	-2.392001	0.083785	Remembering the Utils: Substitutions with Sed
stories/15596311	0.0	-3.173552	0.040173	How Philadelphia Fixed Parking Craters Using Tax Policy
stories/15281593	0.0	-3.278992	0.036299	Show HN: SoundBrake 2.0 – Headphone device alerts you to outside sounds
stories/15277914	0.0	-2.999032	0.047470	Linux Thinkpad Wiki
stories/15427022	0.0	-2.188245	0.100811	With Big Data Comes Big Responsibility
stories/15570022	0.0	-0.797042	0.310659	Show HN: Empathize – manage user conversations easily
stories/15326624	0.0	-2.426735	0.081157	Older versions of Reason not compatible with High Sierra
stories/15275831	0.0	-1.887119	0.131573	Panoply Raises Another $5M for Its Machine Learning Powered Data Warehouse
stories/15593338	0.0	-2.524787	0.074139	Get online payment on your symfony app in half a day
40 rows × 4 columns
The trouble with that is that if we look at the predicted probabilities, most of them are very low, before 0.2.

If I looked only at articles with a predicted probability of 0.5 or above, I wouldn't get many articles at all. Most of the relevant articles have a probability less than 0.5. This is the case because I judge fewer than 10% of the articles to be relevant.
To get a good article feed I will have to set a threshold well under a probability of 0.5; if I do so I will see irrelevant articles, but far fewer than if I tried to look at the whole feed.
The world through my eyes
Currently I am browsing Hacker News with an interface that shows me a selection of top rated articles. As I write this, I have 487 in the queue. For quite a while I have been browsing a random selection of 25% of the articles -- I find that this is around the limit of what I can look at. Thus I pick out the top 120 articles:
487
title
0	How-To: Multi-GPU training with Keras, Python, and deep learning
1	Show HN: Uplink - Build declarative web API clients for Python
2	Making Your First Machine Learning Classifier in Scikit-Learn (Python)
3	Stanford Stats 385: Theories of Deep Learning
4	Show HN: Source-To-Source Debuggable Derivatives in Pure Python
5	Show HN: Cli.fyi – A Potentially Useful Command Line Query Service
6	Feature Visualization: How neural nets build up their understanding of images
7	Retroactive Data Structures (2007) [pdf]
8	UX Mapping Methods Compared: A Cheat Sheet
9	Interfacing Python and C: Advanced “ctypes” Features
10	Juggernaut: Neural Networks in a web browser
11	MediEval is a library that lets you evaluate code in several language runtimes
12	Titan: a new statically-typed and AOT programming language targetting Lua
13	New Draft of Sutton and Barto: Reinforcement Learning (Nov. 5)
14	How many arrays in Python do you know?
15	Tangent: Source-To-Source Autodiff for Python
16	API Star – A smart Web API framework, designed for Python 3
17	Java test library for preparing test data set
18	TensorFlow Neural Network Tutorial
19	A Bag of Tricks for Interperable Machine Learning
20	Show HN: Niched Up – Your job search on steroids
21	Reconciling Semantics and Reflection [pdf]
22	Standardizing Bad Cryptographic Practice [pdf]
23	Show HN: A tiny, markdown based framework for creating API docs
24	Performance comparison between Elasticsearch, InfluxDB and OpenTSDB [pdf]
25	Listen to your code: Problem-solving techniques in programming
26	2017 Litigation Trends Annual Survey: Perspectives from Corporate Counsel [pdf]
27	Show HN: Eleven – A web experiment of AI voice agents and visual interactions
28	Show HN: CheckAnyVIN – Free Automotive Recall Database
29	Deep-spam: Machine Learning to classify email into spam
30	Parity Multi-sig wallets funds frozen (explained)
31	Java interfaces map to Haskell records
32	Basic Structure of a Mod in Arma 3
33	What I learnt building a (simple) self-steering car in TensorFlow
34	Show HN: Designer Lynx – curated links to useful UI design resources
35	Generate a ERM from a PostgreSQL database schema
36	Show HN: A simple JSON schema to understand your data with the help of genson
37	Introducing Serial: Improved Data Serialization on Android
38	A Desktop Poetry Editor Using ReactJS, DraftJS, Canvas API
39	ClojureScript(shadow-cljs): Improved Externs Inference
40	Chromoscope: the Milky Way in different wavelengths
41	Some Rules for Historians (1973)
42	Free video course on evaluating and planning A/B tests using R
43	Internet memes on Ylilauta.org [pdf]
44	Cubetto: A robot teaching kids code and computer programming
45	Microsofts Object Detection Tool has a new home
46	Network protocols, sans I/O, for Python
47	Natural Language Processing with Scala
48	When Data Science Destabilizes Democracy and Facilitates Genocide
49	The DevOps Handbook: Introduction Summary to Everything You Need to Know
50	Show HN: Exploring Relay Modern
51	Stack Overflow Developer Desktop Build – 2017
52	A decade of homelessness: Thousands in S.F. remain in crisis (2014)
53	Show HN: Brook, a cross-platform(Linux/MacOS/Windows/Android/iOS) proxy software
54	TensorFlow r1.4
55	Using neural networks to detect car crashes in dashcam footage
56	Show HN: Showcase by Visual Inspector (Concept Redesign on Popular Sites)
57	Functional Programming Jargon
58	Court demands that search engines and internet service providers block Sci-Hub
59	Making WebGL Dance: How, Where, and What to Draw (2013)
60	Show HN: SolrCloud RDF store to find entity-relations via custom streaming query
61	Why side projects are better marketing tools than blogs
62	Show HN: Diagonal – Hair Transplant Booking Platform
63	Database timeouts
64	Coming Soon: Mentorcruise – A marketplace for tech mentorships
65	Millions of Targets: A Macroscopic Characterization of the DoS Ecosystem [pdf]
66	Code N00b: The Ninety-Ninety Rule and All Your Unfinished Projects
67	Rock–paper–scissors game in less than 10 lines of code
68	Hack Devuan for money (for real)
69	Leveraging the Boundary Between Client and Server in a REST API
70	Art Genius – visual search for art, powered by neural nets
71	Old Versions of Android Apps
72	Having a bad job can be worse for your health than being unemployed
73	Is DevOps Making Life for Data Scientists Better?
74	Decision Laundering as a Service
75	Show HN: An ORM for Python objects, inspired by Django
76	Tasks: Trust but Verify
77	The Problem Roadmap: Only build what matters to customers
78	NumFOCUS: nonprofit for open source scientific computing
79	Show HN: Free and lightweight debugging aid for C#
80	So You Want to Be a Cybersecurity Expert – Careers – Communications of the ACM
81	Graphical tool for OpenStack troubleshooting
82	RemoteStorage: Why bother?
83	How to build FreeBSD/EC2 images (2014)
84	Show HN: Javalin 1.0 – A Kotlin/Java web framework
85	Smart Fabrics: Data Storage and Interaction using Magnetized Fabric
86	AI wastes spammers' time
87	Tool to search replies and quote tweets of a tweet
88	RemoteStorage.js 1.0.0 released
89	Taking KSQL for a Spin Using Real-Time Device Data
90	Continuously Deploying Python Packages to PyPI with CircleCI
91	Artificial Intelligence: 4 New Ways You Can Use It to Drive Sales
92	Show HN: A XMake Integration in IntelliJ IDEA
93	“anyone can kill your contract”
94	This AI Turns Unrecognizable Pixelated Photos into Crystal-Clear Image
95	How to scale daft punk or bootleg ?
96	Generate your one time passwords with the command-line
97	Show HN: Detailed Git status icons in Nautilus
98	The invisible complexity to accomplish a user will
99	A mixed-methods study of meditation-related challenges in Western Buddhists
100	Custom Akka HTTP PathMatcher
101	The limitations of Face ID
102	Adversarial Attack to Capsule Networks
103	Show HN: New Clojure Job Board
104	Gitignore.io Template Fork
105	Blogging Requirements: Top Essencial Tools to Run a Blog (2018 Review)
106	Take Home Programming Interviews Suck
107	Match Nested Brackets with Regex: A New Approach
108	Show HN: Stay fit and win with Outduo
109	Show HN: Introducing Comfygure, a Configuration Manager for Easy Deployments
110	CSS Grid and Grid Inspector in Firefox
111	Cypress testing framework that runs in the browser
112	Understanding linear regression
113	What exactly is IBM doing different with ML?
114	Differential Privacy [pdf]
115	How to Speed Up Massive Data Migration to Amazon S3 – NetApp Blog
116	Macmillan Publishers announced that Pronoun is shutting down
117	URL Encoder
118	Tangent: Source-To-Source Debuggable Derivatives
119	The Mythryl programming language
The ultimate way to evaluate a real-world system is: "does the customer like it?", "what value does it create, etc.". Statisical evaluation is valuable for commercial systems in so far as it supports that goal. Looking at the list above, I'm pretty happy. For instance, in the above sample, Hacker News got swamped with articles about an own goal scored by Ethereum that caused the loss of $280 million dollars. (I could have told them so, but they wouldn't have listened -- thus I'm not interested) A few articles about it show up at the bottom of the list, which I can accept. (Note that many of these are rejected by the filtering that happens before the machine learning model.)
But how well does it work, really?
For many reasons, it is tempting to evaluate performance with a single number. That's important for competitions on Kaggle and ImageNet, but less so in the real world.
As is the case with logistic regression, most classifiers return some kind of confidence value which can be handled with a varying cutoff point that lets us make a trade between getting fewer positive results with higher precision as opposed to more positive results with worse precision.
To evaluate the performance of the system, we need to split up the data set into a training set and a test set. For this experiment, I use a 80% train and 20% test split.
I illustrate the tradeoff by plotting the Receiver Operating Characteristic curve:

This curve illustrates the tradeoff: if I set a high confidence threshold, I am in the lower left corner. Clearly there is a range where I can get 20% or so of the wanted articles with a tiny fraction of unwanted articles. We can get around 60% of the wanted articles with a false positive rate of 20%, and 80% with a false positive around 50%.
Going to upper right, however, there is still a hard core of good articles that will be hard to get without tolerating feed which is mostly irrelevant. That's life. If I read Hacker News without any filter that's what I would get.
Practically one might be concerned with a particular operating point, or, in my case, I feel that I can look at a certain number of postings a day and want to make the most of that. Looking at the curve helps you make choices, but if you want to tweak the algorithm and know if it got better or worse, you still need to distill "quality" down to one number, and that number is the area under (the) curve for the receiver operating characteristic, or the ROC AUC.
For the plot above, the area is 0.76. The diagonal line on the plot represents the case of a classifier without predictive value, and corresponds to an AUC of 0.5.
It's not hard to make a plot that is that bad by making a bad model. For instance, if I use just 1% of the data for training (57 samples) I get a model that hasn't learned anything.

That's an important point, because it takes quite a few judgements to train a text classifier -- that's one reason why text classification hasn't become more widespread (and why Relevance Feedback hasn't set the world on fire.)
I think many people would expect to start getting good results with 5 or 10 judgements, but this system is doing poorly with 50. Real-world success, therefore, involves setting realistic expectations and gathering hundreds, if not thousands, of judgements.
How many judgements does it take to make a good model?
That's a good question, and fortunately scikit-learn gives us the tools to answer it.
The technique is straightforward: I just compute the ROC curves using different amounts of training data. The green curve below is the one to look at, as it represents the AUC ROC for the test data. The red curve represents the AUC ROC for using the training data -- the results look a lot better when we use the training data because the model is overfitting; even though it is learning something relevant to the test data, it's learned how to cheat on the training data, finding words that can identify and classify most of the individual titles in the data set:

I don't know exactly how to map the ROC AUC to how satisified I would be with the output, but looking at the error bars, I'd imagine that I'd be happy with the results if I had 1200 training samples; maybe even 500. On the other hand, the AUC seems to be reaching an asymptope around 0.75 and it seems likely that I wouldn't get dramatically better results by adding more training data to this model.
I zoom in on the case of low numbers of training samples in order to get a sense of where learning starts:

The error bars seem to clearly be pulling away from the 50% mark when we have somewhere between 100 to 200 examples. Using 2% of the data for training (115 samples) the ROC curve is visibly pulling away from the diagonal:

The user's subjective experience is the real metric here, so if I look at the sorted predictions with 115 training samples, they really aren't bad:
prediction	log-odds	probability	title
story				
stories/15451601	1.0	0.934137	0.717914	An algebraic generalization for graph and tensor-based neural networks [pdf]
stories/15425829	1.0	0.880493	0.706924	Deep Learning Library 1.0 – Fast Neural Network Library
stories/15441782	1.0	0.795858	0.689088	Stanford Network Analysis Platform: Graph Mining Library and Datasets
stories/15629253	1.0	0.604679	0.646726	Show HN: PySchemes – A library for validating data structures in Python
stories/15374307	1.0	0.411127	0.601358	FastText: Library for efficient text classification and representation learning
stories/15558018	1.0	0.385377	0.595169	TimescaleDB: PostgreSQL made scalable for time-series data [pdf]
stories/15383148	1.0	0.294462	0.573088	Learning from Feedback: Evidence from New Ventures [pdf]
stories/15389287	1.0	0.232262	0.557806	Libact: Pool-Based Active Learning in Python
stories/15636730	1.0	0.218231	0.554342	Neural Discrete Representation Learning [pdf]
stories/15457181	1.0	0.148527	0.537064	Show HN: Scikit-plot – One-line functions for detailed visualizations – Python
...	...	...	...	...
stories/15607951	0.0	-3.895816	0.019922	Powering an Indoor Cycle Studio with Our Homegrown Technology, AKA, the Cyclers
stories/15423808	0.0	-3.902088	0.019800	Google, how old is the president?
stories/15493285	0.0	-3.910505	0.019637	Old Games, Remasters, and the Joy of Owning Your Work
stories/15480900	0.0	-3.960173	0.018703	When the plane takes off, why doesn’t the tail hit the runway?
stories/15394341	0.0	-3.984242	0.018267	The Gun Lobby Is Down to Its Last, Unconvincing Excuse
stories/15558175	0.0	-3.991387	0.018139	The universe shouldn’t exist, according to science
stories/15278500	0.0	-4.034383	0.017389	The Psychology of iPhone:A Neuromarketing Strategy, or an Overdose?
stories/15476574	0.0	-4.298847	0.013402	Bruno Latour, a veteran of the ‘science wars,’ has a new mission
stories/15284741	0.0	-4.546106	0.010497	'Bitcoin is a bubble,’ says the head of the world’s largest hedge fund
stories/15360671	0.0	-4.574913	0.010202	If you made a sphere of all the earth’s water, how big would it be?
5656 rows × 4 columns
Looking at the graphs, it seems there is a threshold between 50 and 100 training examples where the results go from "artificial stupidity" to "artificial intelligence": at 57 samples, the model only seems to have learned that it likes pdf files:
prediction	log-odds	probability	title
story				
stories/15283998	0.0	-0.335823	0.416824	AI-Produced Journalism â How WaPo's robo reporter published 850 articles
stories/15620794	0.0	-0.447031	0.390067	Multivariate Dependence Beyond Shannon Information [pdf]
stories/15278498	0.0	-0.660584	0.340608	Show HN: Peer review published journal articles
stories/15294773	0.0	-0.686719	0.334763	Traceroute and BGP AS Path Incongruities [pdf]
stories/15372054	0.0	-0.686719	0.334763	BIND 9.12 refactoring and performance improvements [pdf]
stories/15305161	0.0	-0.686719	0.334763	Zcash ceremony and audit [pdf]
stories/15354064	0.0	-0.689089	0.334236	Simple, Fast and Safe Manual Memory Management [pdf]
stories/15322398	0.0	-0.717672	0.327906	Mozilla's Internet health report [pdf]
stories/15341088	0.0	-0.723023	0.326728	OpenSMTPD, current state of affairs [pdf]
stories/15579474	0.0	-0.729435	0.325319	Non Serviam [1970, pdf]
...	...	...	...	...
stories/15588822	0.0	-3.581264	0.027086	Why is PyPy for python not as popular in the community?
stories/15328007	0.0	-3.614859	0.026215	Ask HN: Is it ethical to ask me to write software for free before an interview?
stories/15278718	0.0	-3.618457	0.026123	The Problems with Redux: Are MobX and Realm Going to Put and End to It?
stories/15273416	0.0	-3.672061	0.024794	Google’s Tez payments app just went live. We tried it out to see how fast it is
stories/15319224	0.0	-3.679032	0.024626	Org-Mode Is One of the Most Reasonable Markup Language to Use for Text
stories/15339434	0.0	-3.687560	0.024422	A Notice by the Homeland Security (to collect social media info for immigrants)
stories/15315055	0.0	-3.835584	0.021133	Lynx identity manager is coming out of the garage. Want to take it for a spin?
stories/15289372	0.0	-3.841080	0.021019	What is the most confusing thing to you in Python?
stories/15461825	0.0	-3.915131	0.019548	Why is animation the medium of choice for startups to explain what they do?
stories/15601103	0.0	-4.265431	0.013851	The transition from mobile-first to AI-first is not only for Google but it’s
5714 rows × 4 columns
The consolation prize is that this model knows that it sucks (something good about logistic regression.) It never predicts that a posting has more than a 50% chance of being relevant, so if we set the decision threshold of 50%, it predicts that everything is irrelevant.
What about other features?
One ground rule of this analysis is that I'm only using information available from the Hacker News link, which includes the title, the submitter, and the url. We've seen that the title is fairly predictive.
What about the user?

I would have thought that I would have some 'friends' and 'foes' on HN, but it seems like the submitting user is not at all predictive of relevance.
To double check, here are some of the predictions.
prediction	log-odds	probability	title
story				
stories/15636853	0.0	-0.117590	0.470636	Resonance Audio: Multi-platform spatial audio at scale
stories/15346449	0.0	-0.257229	0.436045	Churn Prediction with Automatic ML
stories/15355051	0.0	-0.384499	0.405042	Acebusters opensourced own web app
stories/15528458	0.0	-0.512598	0.374585	Day 4 of 100 Days of VR: Going Through the Unity Space Shooter Tutorial III
stories/15528388	0.0	-0.512598	0.374585	Record proportion of people in employment are home workers (UK, 2014)
stories/15276421	0.0	-0.563047	0.362843	Show HN: TinyMCE and Evernote and RegEx Search
stories/15276386	0.0	-0.563047	0.362843	Creating a SaaS the wrong way
stories/15276417	0.0	-0.563047	0.362843	Optionsbleed bug in Apache HTTPD
stories/15633591	0.0	-0.570212	0.361188	Inventive Coding font in beta testing: CSS Mono
stories/15635154	0.0	-0.638260	0.345640	JUCE – Mobile Audio Quality Index
...	...	...	...	...
stories/15288654	0.0	-2.953646	0.049564	Lifelike Robots Possible with New “3x Stronger Than Human” Synthetic Soft Muscle
stories/15288703	0.0	-2.953646	0.049564	Show HN: Vincent – affordable, automatic meeting scheduling
stories/15288687	0.0	-2.953646	0.049564	German hackers step in to fix voting machines
stories/15358471	0.0	-2.998621	0.047488	Spending Time at India's First Mobile Congress
stories/15273464	0.0	-2.998621	0.047488	Is Social Media a Failure?
stories/15439743	0.0	-2.998621	0.047488	5 preload gotchas i wish i would knew in advance
stories/15273481	0.0	-2.998621	0.047488	Top Rated Code Editors for Developers
stories/15273338	0.0	-2.998621	0.047488	DSC with Infrastructure-As-Code and Azure Automation is a potent combination
stories/15577421	0.0	-2.998621	0.047488	Learning Depth-Three Neural Networks in Polynomial Time
stories/15273416	0.0	-2.998621	0.047488	Google’s Tez payments app just went live. We tried it out to see how fast it is
1155 rows × 4 columns
Those are pretty bad. In contrast, I get much better results using url hosts as a feature. (Like the badhosts mechanism used in the prefilter)

Looking at the RoC curve it seems that the hostname is highly predictive at the high-relevance and low-relevance sides of the curve, probably because there are a few sites that host a large number of articles which are balanced towards one direction or the other. Here are the top and bottom predictions:
prediction	log-odds	probability	title
story				
stories/15282816	1.0	0.269928	0.567075	One Leak Will Sink a Ship: WebRTC IP Address Leaks
stories/15553791	1.0	0.269928	0.567075	A Cost-Benefit Approach to Improving Browser Security
stories/15577421	1.0	0.269928	0.567075	Learning Depth-Three Neural Networks in Polynomial Time
stories/15607450	1.0	0.269928	0.567075	How intelligent are convolutional neural networks?
stories/15273210	1.0	0.269928	0.567075	Mathematics via Symmetry (2013)
stories/15372615	1.0	0.269928	0.567075	Infinitely many twin primes
stories/15488202	1.0	0.269928	0.567075	Generalization in Deep Learning [pdf]
stories/15414046	1.0	0.269928	0.567075	Interpretable Convolutional Neural Networks
stories/15626053	1.0	0.269928	0.567075	Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles [pdf]
stories/15567373	0.0	-0.430549	0.393995	Can Vector Supercomputing Be Revived?
...	...	...	...	...
stories/15304986	0.0	-2.584873	0.070118	Project Jengo Strikes Its First Targets (and Looks for More)
stories/15404615	0.0	-2.626042	0.067481	Jayant Baliga's Quest to Make Silicon-Carbide Power Electronics Succeed
stories/15401254	0.0	-2.626042	0.067481	Three Advances Make Magnetic Tape More Than a Memory
stories/15400232	0.0	-2.626042	0.067481	China Demonstrates Quantum Encryption by Hosting a Video Call
stories/15438594	0.0	-2.626042	0.067481	GM Cruise Snaps Up Solid-State Lidar Pioneer Strobe Inc
stories/15384044	0.0	-2.626042	0.067481	AI to Help Power Grids Resist Disruptions
stories/15344022	0.0	-2.626042	0.067481	Drone with Event Camera Takes First Autonomous Flight
stories/15548852	0.0	-2.626042	0.067481	The Lure of a Fully Decentralized Internet
stories/15492303	0.0	-2.664825	0.065081	Turn your laptop into a portable whiteboard
stories/15454432	0.0	-2.664825	0.065081	Announcing Deep Learning, No Coding (DLNC)
1155 rows × 4 columns
Although the ROC AUC looks pretty good (better than I got for 117 training examples using words;) the top and bottom don't look so good to me.
Since we do seem to get some value out of the hostnames, what if we use the title words, submitters and hostnames to make one big model? The AUC is a touch better than what we get from using just the words (0.77 vs 0.74)

prediction	log-odds	probability	title
story				
stories/15528326	1.0	3.777827	0.977639	Study notes: Why deep and cheap learning works
stories/15527967	1.0	3.424605	0.968465	Learning Diverse Skills via Maximum Entropy Deep Reinforcement Learning
stories/15277179	1.0	3.032455	0.954019	Real-time object detection and deep learning and Python
stories/15451334	1.0	2.475431	0.922401	Mixed Precision Training for Deep Learning Models
stories/15438279	1.0	2.428966	0.919010	Neural Turing Machines (NTM): PyTorch Implementation
stories/15305774	1.0	2.409528	0.917551	Visualizing Dataflow Graphs of Deep Learning Models in TensorFlow
stories/15567545	1.0	2.288442	0.907915	A Deep Reinforcement Learning Framework for Crypto Financial Portfolios
stories/15461477	1.0	2.167837	0.897324	Show HN: Python library to automate your time management in Airtable and others
stories/15608012	1.0	1.943615	0.874749	DeepXplore: automated whitebox testing of deep learning systems
stories/15350422	1.0	1.874672	0.866998	Deep Learning in Finance [pdf]
...	...	...	...	...
stories/15307503	0.0	-5.208870	0.005438	5 Things You Should Look for in a Web Development Company
stories/15498010	0.0	-5.379586	0.004589	How Dopamine Tells Us Something Isn’t Worth the Wait
stories/15415392	0.0	-5.392626	0.004529	2017 Nobel Peace Prize to the International Campaign to Abolish Nuclear Weapons
stories/15294274	0.0	-5.452892	0.004266	Medical Device Security – Current State of Play
stories/15407336	0.0	-5.493184	0.004098	Google's Pixel Buds translation will change the world
stories/15504074	0.0	-5.662114	0.003463	Learn Go Programming – Monthly Recap #1
stories/15363571	0.0	-5.666140	0.003449	Nvidia CEO says Moore’s Law is dead and GPUs will replace CPUs
stories/15273129	0.0	-6.020664	0.002422	‘Tez’: Google’s new digital payment app for India
stories/15288087	0.0	-6.024575	0.002413	Here Is the Pixel 2 in “Kinda Blue,” White, and Black – Starting at $649
stories/15439686	0.0	-6.324011	0.001790	Security for People Who Don’t Want to Spend Much Time Thinking About Security
1155 rows × 4 columns
Next time I run my model (probably this evening) I will switch to the joint model since it seems to perform a bit better.
Conclusion
Trained on roughly 5000 examples, a simple text classification model based on words in the title has proved effective for improving the relevance of a feed of articles from Hacker News. (See my last essay for my definition of "relevance".)
A model based on the submitter of the article was completely ineffective, and a model based on the hostname where the article is based was less effective than the word-based model. Adding both of those features improves relevance a bit on the ROC AUC (Reciever Operating Characteristic Area Under Curve) metric.
Although I might be able to do better by tweaking the features or using a different classifier than Logistic Regression, I am not inclined to mess with this model much more because a model based on the title and hyperlink is missing a large amount of information which could only be found on the target webpage. I'll be looking to that for future improvements, but that is going to involve beefing up the information harvesting system and probably some performance work on the machine learning side since the amount of text will increase greatly.
The one area where I did explore the parameter space was in the size of the training set. Particularly, this system seemed to learn nothing of value when given 57 training examples but it did visibly seem to learn when given 114 examples. Although the results improve when we use more training examples, there does seem to be a minimum number examples required to get meaningful results. Without some major change in the algorithm, therefore, classifiers that want to get across the "commercialization valley of death" will need to coax users to classify more documents than they might want to initially.