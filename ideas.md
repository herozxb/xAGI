# 1 overfitting like memorizing, fitting is learning

# 2 do the simple concept of AI really well, then the bigger task can be mastered

# 3 NeRF https://keras.io/examples/vision/nerf/

# 4 learning and thinking about from the papers

# 5 Model Design Embeds Inductive Biases
Inductive biases are assumptions baked into a model about the nature of the data. These biases help guide the learning process, allowing the model to generalize well to unseen data.

Inductive Bias in CNNs: CNNs assume spatial locality — that nearby pixels are more correlated than distant ones — which is generally true for images. This bias enables CNNs to focus on local features, making them more data-efficient.
Inductive Bias in Transformers: Transformers assume that every part of a sequence might need to relate to every other part, which is true for language. Their structure inherently supports the kind of flexible, non-local attention that is ideal for language, audio, and other sequence data.
These biases make learning faster, reduce the amount of data needed, and improve generalization.
# 6 Efficiency in Learning and Computation
This is why the model’s architecture often outperforms purely end-to-end approaches that don’t embed structural knowledge or focus on computational efficiency.
# 7 why the CNN add 3x3 filter and Transformor add attention QKV works, why the biase of CNN and Transformer works
Parameter Reduction: Smaller filters have fewer parameters compared to larger ones (like 5x5 or 7x7), reducing computational cost and the risk of overfitting.
Overcoming Limitations of RNNs, Similarity to Search Mechanisms, In information retrieval, a query is used to search a database of documents (keys) to retrieve relevant information (values).
