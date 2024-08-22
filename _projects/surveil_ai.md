---
layout: page
title: Surveilai
description: Advanced Semantic Search for Surveillance Footage
img: assets/img/surveliance.png
importance: 2
category: AI
---

In today’s world, surveillance systems are a critical component of security, generating vast amounts of video footage daily. However, analyzing this data manually is not only time-consuming but also prone to errors, making it challenging to extract valuable insights quickly and efficiently. The need for a more effective solution has led to the development of a cutting-edge image captioning system that leverages deep learning techniques to automate the process of analyzing surveillance footage.


<h2>The Challenge of Surveillance Data</h2>
Surveillance systems around the globe produce an overwhelming volume of video data every day, creating a significant challenge for security teams and forensic investigators. Traditional methods of reviewing this footage are labor-intensive, requiring analysts to manually sift through hours of video to identify relevant events. This process is not only slow but also susceptible to human error, with critical details easily overlooked due to the sheer volume of data.

To address these challenges, this experiment aimed to develop an advanced solution that automates the analysis of surveillance footage. By using state-of-the-art image captioning and semantic search techniques, this system can generate descriptive captions for video frames, enabling quick and accurate identification of important events.


<div class="row justify-content-sm-center">
    
<div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.html path="assets/img/713_arch.png" title="Solution Architecture" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">Solution Architecture</div>


<h2>Methodology: Building the Image Captioning System</h2>
The success of this image captioning system hinges on a meticulously designed methodology that integrates powerful deep learning models with advanced data processing techniques. Below, the core components of this approach are detailed. 


<h3>Data Preparation: The MS COCO 2017 Dataset</h3>
The foundation of any deep learning model lies in the quality of its training data. For this work, the <a href="https://cocodataset.org/#home"> MS COCO 2017</a> dataset was selected, a comprehensive resource widely used in computer vision tasks. This dataset includes over 200,000 images with multiple human-generated captions, providing a rich and diverse set of examples for training the model. The diversity of the dataset, encompassing various objects, scenes, and activities, ensures that the model can generalize well to real-world surveillance footage.


<div class="row justify-content-sm-center">
    
<div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.html path="assets/img/img_proc.png" title="Image Preprocessing" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">Image Preprocessing</div>


<h3>Model Architecture: A Dive into the System’s Components</h3>
The core of this image captioning system is built on a combination of convolutional neural networks (CNNs) and transformers, two of the most powerful tools in deep learning.

<div class="row justify-content-sm-center">
    
<div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.html path="assets/img/img_captioning_m.jpg" title="Image Captioning Model" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">Image Captioning Model</div>

<h4>Pretrained Image Models: ResNet152 and InceptionV3</h4>
This experiment began by using pretrained CNNs— InceptionV3 and ResNet152 —to extract detailed visual features from each image. These models, trained on the ImageNet dataset, are renowned for their ability to capture intricate patterns and structures within images.

<ul>
    <li><a href="https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html">ResNet152</a></li><p>This model is equipped with 152 layers, allowing it to learn deep and complex features. The use of residual connections in ResNet152 mitigates the vanishing gradient problem, which is critical for training deep networks effectively.</p>
    <li><a href="https://pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights">InceptionV3</a></li>
    <p>Known for its efficiency, InceptionV3 employs a combination of convolutions with different filter sizes, enabling it to capture multi-scale features. This makes it particularly effective for understanding diverse image contents in surveillance footage.</p>
</ul>

<h5>Transformer Encoder-Decoder Model</h5>
Once the visual features are extracted by the CNNs, they are passed to a transformer-based encoder-decoder model. The transformer architecture, celebrated for its self-attention mechanisms, is well-suited for sequence-to-sequence tasks. In this system, the transformer processes the image embeddings (sequences of numerical data representing the images) and generates descriptive text sequences (captions) that convey the content of the frames.

<div class="row justify-content-sm-center">
    
<div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.html path="assets/img/transformer_arch.jpg" title="Transformer (Encoder-Decoder) Model Architecture" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">Transformer (Encoder-Decoder) Model Architecture</div>


The encoder component of the transformer focuses on understanding the spatial and contextual relationships within the image data. The decoder then uses this information to produce coherent, contextually relevant captions. The inclusion of multi-head self-attention mechanisms ensures that the model can simultaneously consider different parts of the image, leading to more comprehensive and accurate descriptions.

<h5>Byte Pair Encoding (BPE) Tokenizer</h5>
To effectively handle the vast and varied textual data generated by the model, this experiment implemented the Byte Pair Encoding (BPE) tokenizer. BPE is a subword segmentation technique that plays a crucial role in managing vocabulary, particularly in tasks involving natural language processing (NLP), such as image captioning.

<i>Why BPE?</i><br/>
Traditional tokenization methods, which break text into words, often struggle with handling rare words or out-of-vocabulary (OOV) terms. In surveillance scenarios, where the system might encounter uncommon objects, names, or actions, the ability to accurately tokenize and understand these terms is vital. BPE addresses this by breaking down words into smaller, more manageable subword units.

<i>How BPE Works</i><br/>
BPE begins by treating each word as a sequence of individual characters. It then iteratively merges the most frequent pairs of characters or subwords into larger subword units. For instance, the word “playing” might initially be split into individual characters like `“p,” “l,” “a,” “y,” “i,” “n,” “g.”` BPE would then merge the most frequent pairs, eventually forming meaningful subwords like “play” and “ing.” This process continues until a predefined vocabulary size is reached. <a href="https://pvsnp9.github.io/blog/2024/embeddings/"> Explore Tokenization </a>


By integrating BPE into the image captioning pipeline, this experiment significantly improved the model’s ability to generate accurate and contextually appropriate captions, even in challenging surveillance scenarios.


<h2>Experiments: Refining and Evaluating the System</h2>

The development of this image captioning system involved rigorous experimentation to fine-tune the model architecture and optimize its performance. Here, the key experiments that shaped the final model are outlined.

<h3>Experiment Zero: InceptionV3 to Decoder Model</h3>
The first experiment explored the use of InceptionV3 as the image encoder in the captioning model. Initially, the encoder output was configured to produce embeddings with a shape of (batch_size, d_model). However, during validation, it was observed that the model struggled to generalize across different images, often producing similar captions for diverse inputs.

To address this issue, the architecture was adjusted so that the encoder generated sequences rather than single embeddings. This modification allowed the model to capture more comprehensive details about each image, improving its ability to generate varied and accurate captions. Additionally, a warm-up decay learning rate mechanism was implemented to stabilize the training process. This adjustment further enhanced the model’s performance, resulting in more precise and contextually appropriate captions.

<h3>Experiment One: ResNet152 to Fully Encoder-Decoder Model</h3>

In the second experiment, the ResNet152 model was integrated into a fully encoder-decoder architecture. This configuration included a custom loss function designed to ensure that the model’s attention was evenly distributed across the entire image, preventing any part of the image from being neglected.
```json
{
  "major_configuration": {
    "Image encoder dropout": [0.3, 0.5],
    "Enc-dec number of layers or blocks": [4, 6, 8],
    "Embedding size (d_model)": 512,
    "Number of heads": 8,
    "Sequence length": [81, 144, 196],
    "Enc-dec dropout": [0.1, 0.2],
    "Adaptive learning rate": "10^-5 - 10^-3",
    "Gradient clipped to": "5 (max)",
    "Epochs": 50,
    "Batch size": [32, 48, 64, 128],
    "Number of trainable parameters": "136M+"
  }
}
```


During the initial epochs, some issues were encountered with the gradient L2 norms, which dropped significantly—a potential indication of gradient vanishing or poor learning dynamics. However, by carefully adjusting the learning rate and continuing the training process, the model was stabilized. As training progressed, the model began to generate more accurate and detailed captions, showcasing its ability to understand and describe complex scenes in surveillance footage.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/attention_1.gif" title="Attention scores" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/attention_2.gif" title="Attention scores" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/attention_3.gif" title="Attention scores" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A Encoder-Decoder cross attention map with text and images (gifs). 
</div>


To evaluate the model's performance, standard metrics commonly used in natural language processing and computer vision were employed:
<ul>
    <li>BLEU (Bilingual Evaluation Understudy):<span>Measures the precision of n-grams between the generated and reference captions.</span></li>
    <li>GLEU (Google-BLEU): <span>Balances precision and recall for a more comprehensive assessment of caption quality.</span></li>
    <li>METEOR (Metric for Evaluation of Translation with Explicit ORdering): <span>Considers precision, recall, and synonyms, providing a robust evaluation of the generated captions.</span></li>
</ul>

These metrics provided a detailed understanding of how well the model was performing, guiding further optimizations.


<h3>Enhancing Search with Vector Databases and Semantic Search</h3>

While generating accurate captions is a critical component, the ability to efficiently search and retrieve relevant frames from vast amounts of video data is equally important. To achieve this, the work incorporated vector databases and semantic search techniques into the system.


<h4>Vectorizing and Storing Data</h4>
After generating captions and other relevant metadata, an efficient way to store and query this information was needed. A vector database was used to store both sparse and dense embeddings of the textual data. Sparse embeddings, generated using the BM25 encoder, are ideal for keyword-based searches, capturing the importance of specific terms. Dense embeddings, on the other hand, are produced using models like ClipViT (Vision Transformer) and enable semantic searches by capturing the context and meaning of the text.

Each frame's text extraction is stored as a vector in a scalable vector database. By vectorizing the captions and metadata, this system can efficiently perform similarity searches, retrieving frames that are contextually related to a user’s query, even if the exact keywords are not present.

<h4>Semantic Search: A Hybrid Approach</h4>
To enable robust search capabilities, a hybrid search approach that combines both sparse and dense embeddings was implemented. This allows users to perform sophisticated queries that leverage both exact keyword matching and semantic understanding.
<ul>
<li>Sparse Embeddings: <span>These are particularly effective for traditional keyword searches, where users need to find frames containing specific terms or phrases.</span></li>
<li>Dense Embeddings:<span>These enable semantic searches, allowing the system to understand the context and retrieve frames that are relevant even if the query terms are not explicitly mentioned in the captions.</span></li>
</ul>

For example, a user searching for "person wearing a red jacket" might retrieve frames not only where those exact words are mentioned but also where similar concepts are inferred from the context, thanks to dense embeddings.

<h4>Vector Database: Storage and Retrieval</h4>
The vector database, managed through Pinecone, is optimized for handling high-dimensional vector data, making it an ideal solution for this project’s needs. Pinecone’s robust indexing and search capabilities ensure that even as the volume of data grows, the system remains fast and responsive.

By integrating vector databases and semantic search into the image captioning system, the ability to retrieve relevant frames quickly and accurately was significantly enhanced, further improving the efficiency of surveillance data analysis.


<h3>Future Directions</h3>
While this image captioning system has shown strong performance, there are opportunities for further improvement. Future work could focus on refining the attention mechanisms within the model or exploring alternative architectures that might enhance the quality of the generated captions even further. Additionally, expanding the training dataset to include more varied surveillance footage could improve the model’s ability to generalize to different real-world scenarios.


In conclusion, the integration of deep learning and semantic search techniques into surveillance analysis offers a powerful tool for improving the efficiency, accuracy, and usability of video data. As these technologies continue to evolve, they are poised to play an increasingly important role in enhancing security and investigative operations worldwide.

<h3>Acknowledgment</h3>
I would like to express my gratitude to <a href="https://twitter.com/satilame">Sathyajit Loganathan</a> for his contribution to this experiment. 

<h3>References</h3>

1. Cornia, M., Baraldi, L., Serra, G., & Cucchiara, R. (2020). Meshed-Memory Transformer for Image Captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10578-10587.

2. Rennie, S. J., Marcheret, E., Mroueh, Y., Ross, J., & Goel, V. (2017). Self-Critical Sequence Training for Image Captioning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 7008-7024.

3. Anderson, P., Fernando, B., Johnson, M., & Gould, S. (2016). SPICE: Semantic Propositional Image Caption Evaluation. In Proceedings of the European Conference on Computer Vision (ECCV), 382-398.

4. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. In European Conference on Computer Vision (ECCV), 740-755.

5. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is All You Need. In Advances in Neural Information Processing Systems (NeurIPS), 5998-6008.

6. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), 4171-4186.

<h3>Appendix</h3>



<div class="row justify-content-sm-center">
    
<div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.html path="assets/img/enc_dec_spatial_channels_batch_1.png" title="Encoder Output to Decoder" class="img-fluid rounded z-depth-1" %}
</div>
</div>
<div class="caption">Head and Channelwise attention from Encoder to Decoder (Cross attention). The output from vision model that is fed through the encoder model.</div>