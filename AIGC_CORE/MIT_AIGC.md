# 1 	Autoencoder || x - x^ ||^2 
# 1.1	image to image
# 1.2 	GPT text to text 
# 1.3 	VAE, Variational autoencoder are a probabilistic twist on autoencoder 
# 1.4	regularization and the normal prior, continuity and completeness
# 1.5   Deep Fake, two VAE, two encoder, two decoder, encoder one to decoder two
# 1.6   VAE, Stable Diffusion, Open-Sora-Plan, U-net, are all Autoencoder
# 1.7 	uncolor image to color image


# 2 	GAN





# 3 	fundation model
# 3.1 	transformer

# 4 	training data
# 4.1 	garbage in, garbage out
# 4.2 	the model does not seen, it is random in that unseen region, car crash


# 5	Neural network limitation

# 5.1	Very data hungry
# 5.2   Computationally intensive to train and deploy
# 5.3 	Easily fooled by adversarial examples
# 5.4	Can be subject to algorithmic bias
# 5.5   Poor at representing uncertainty ( how do you know what the model knows )
# 5.6	Uninterpretable black boxes, difficult to trust
# 5.7	Often require expect knowledge to design, fine tune architectures
# 5.8   Difficult to encode structure and prior knowledge during learning
# 5.9	Extraploation struggle to go beyond the data

# 6 	Encode structure into deep learning
# 6.1   CNNs : Using Spatial Structure
# 6.2	GCNs : Graphs as a Structure for Representing Data, Molecular Discovery, Traffic Prediction,COVID-19 Forecasting 
# 6.3   Logic : Tree structure into deep learning



# 6.4   Attention : Matrix structure into transformer


# 7 	Diffusion model
# 7.1   diffusion generate protein



# 8 	Generative model

	Autoregressive Models (ARMs): ARMs parameterized with Causal Convolutionas and Transformers
	Flow-based models (flows): RealNVP and IDFs (Integer Discrete Flows)
	Variational Auto-Encoders (VAEs): a plain VAE and various priors, a hierarchical VAE
	Diffusion-based Deep Generative Models (DDGMs): a Gaussian forward diffusion
	Score-based Generative Models (SBGMs): score matching, score-based generative models, (conditional) flow matching
	Hybrid modeling
	Energy-based Models
	Generative Adversarial Networks (GANs)
	Neural Compression with Deep Generative Modeling
	
# 9	attention is all you need
# 9.1 	in order to predict the future, we need to understand the past,  S = Where are we [ going ]

# 10  	Chromosome, genome
# 10.1 understand all the genomes, how they works, protein structure prediction 
# 10.2 you need very good at creat the creature by the genome you learn, when you want to do longivity

# 11 	p( data | theta )
# 11.1  the generate result is the data is 2D image, the theta is given prompt with other paramenters, so the output of the NN is 2D image the z to output of AutoEncoder
# 11.2 we train the p( data | theta ) by given data to get the theta, the ideal way is the theta is for all the data, but the PC is limited, so we use batch to get the theta which is for batch
# 11.3 then we use the theta we trained to generate the data, when the data is 2D image, then the output of NN is (1024*1024) image, and the theta is text made token or noise image

# 12 	Negative log likelihood with MSE
# 12.1 N( y; y^hat(x;w), delta^2 ), the NN is y^hat( x; w ), w is the NN parameter
# 12.2 sum( log( N( y; y^hat(x;w), delta^2 ) ) ) = log( N( y; y^hat(x;w), delta^2 ) * N( y; y^hat(x;w), delta^2 ) * ... )

# 13 	bayes p(theta|data) = p(data|theta) * p(theta) has p(data|theta) in it, and p(data|theta) is for the AIGC, that is given theta we generate the data, but AIGC is not use bayes  


# 14 	generative AI, self supervise learning
# 14.1	GPT is self supervise learning			Text input 	=> Text input 	(same)
# 14.2	AutoEncoder is self supervise learning		Image		=> Image	(same)
# 14.3	Diffustion is kind of self supervise learning		Image		=> Image	(same)

# 15	MEANING of word
# 15.1	MEANING is relation
# 15.2 MEANING in an embedding space
# 15.3	represented by multiple digits
# 15.4	context depended, river bank, finicial bank
# 15.5 need to embed sequences of word, context meaning, pre-training vs downstream task, pre-training as useful as possible, simulation vs reality
# 15.6 Meaning is defined by the company it keeps, high dimensional embedding spaces
# 15.7 meaning of    word
			|
		something that is said
		
# 16 	predict next pixel
# 17 	game of compression,	image > z < image, autoencoder, George Hotz
# 18 	Plug and play
	image > (embed) z (generate) < image
	text  > (embed) z (generate) < text
	audio > (embed) z (generate) < audio
	
	
# 19 	"make me a sandwich" -> robot brain -> embeding ( unknow number ) -> doing something
					( go to kichen, take bread from the pantry, ..... )
				( robot brain of LLM ) => ( go to kichen, take bread from the pantry, ..... )

# 20 	automate agent, planning, thinking, reasoning
	analyze -> generate ->  text file -> nalyze -> generate ->  text file -> nalyze -> generate ->  text file -> ......
	[I] analyze the problem, [I] make a question -> chatgpt generate -> text file -> [I] analyze the problem, [I] make a question -> chatgpt generate -> text file -> [I] analyze the problem, [I] make a question -> chatgpt generate -> text file -> [I] analyze the problem, [I] make a question -> chatgpt generate -> text file -> ......
	analyze -> tools -> generate ->  text file -> analyze  -> tools  -> generate ->  text file -> analyze  -> tools  -> generate ->  text file -> ...... tools ( internet, calculators, other foundation models ) 

# 21 	market Winner takes all
