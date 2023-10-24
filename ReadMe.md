## Introduction to Explainability with Captum

The interpretability of AI is the ability of describing AI models in human understandable terms. By understanding AI models better and why they are making certain predictions, we can start to answer difficult questions about the internals of our models, trust, accountability, and fairness.

*Captum* is a model interpretability library for PyTorch which currently offers a number of attribution algorithms that allow us to understand the importance of input features, and hidden neurons and layers. 

Captum supports any PyTorch model, meaning it is not limited to classification models but can also be used for any application or domain. Some of the algorithms might be more common for certain types of applications, such as computer vision.

Captum provide libraries for implementing both LIME and SHAP and also others attribution algorithm.

The diagram below shows all attribution algorithms available in the Captum library divided into two groups. The first group, listed on the left side of the diagram, allows us to attribute the output predictions or the internal neurons to the inputs of the model. The second group, listed on the right side, includes several attribution algorithms that allow us to attribute the output predictions to the internal layers of the model. 

<center>  <img src="https://drive.google.com/uc?export=view&id=1Y_kly2vU15h7F0WY2HOKRBQFemXJljBr" width="950" height="400"> </center> 



##Interpreting MLP with LIME

LIME is an algorithm that can explain the predictions of any classifier or regressor in a faithful way, by approximating it locally with an interpretable model.

Check ["Why should i Trust You - Explaining the prediction of any classifier"](https://arxiv.org/pdf/1602.04938.pdf) for a better understanding 

When we build an *Explainer* generally we want it to have some characteristics. 
- **Interpretability**: this is an essential criterion for explainations. Interpretability is the ability to provide quanlitative understanding between the input varaibles and the response.
- **Local Fidelity**: even though it's almost impossible for an explainer to be completely faithful, unless it is the complete description of the model itself, foran explanation to be meaningful it must at least be *locally faithful*, i.e. it must correspond to how the model behaves in the vicinity of the instance being predicted.
- **Model-Agnostic**: While there are models that are inherently interpretable an explainer should be able to explain any model, and thus be model-agnostic (i.e. treat the original model as a black box). 

**LIME** (Local Interpretable Model Agnostic Explainer) is an algorithm that allows to identify an interpretable model over the interpretable representation that is locally faithful to the classifier. 

### Interpretable Representation

Firstly, it is important to distinguish between features and interpretable data
representations. As mentioned before, interpretable explanations need to use a representation that is understandable to humans, regardless of the actual features used by the model. For example, a possible interpretable representation
for text classification is a binary vector indicating the presence or absence of a word, even though the classifier may use more complex (and incomprehensible) features such as word embeddings. Likewise for image classification, an interpretable representation may be a binary vector indicating the “presence” or “absence” of a contiguous patch of similar
pixels (a super-pixel), while the classifier may represent the
image as a tensor with three color channels per pixel



### LIME with Captum

There are 2 implementations of LIME in caputm. The first is called simply LIME.

Essentially, Lime trains an **interpretable surrogate model** to simulate the target model's predictions. So, building an appropriate interpretable model is the most critical step in Lime. Fortunately, Captum has provided many most common interpretable models to save the efforts. We will demonstrate the usages of Linear Regression. Another important factor is the similarity function. Because Lime aims to explain the local behavior of an example, it will reweight the training samples according to their similarity distances. By default, Captum's Lime uses the exponential kernel on top of the consine distance.

Lime **assumes that the interpretable representation is a binary vector**, corresponding to some elements in the input being set to their baseline value if the corresponding binary interpretable feature value is 0 or being set to the original input value if the corresponding binary interpretable feature value is 1. Input values can be grouped to correspond to the same binary interpretable feature using a feature mask provided when calling attribute, similar to other perturbation-based attribution methods.

One example of this setting is when applying Lime to an image classifier. Pixels in an image can be grouped into super-pixels or segments, which correspond to interpretable features, provided as a feature_mask when calling attribute. Sampled binary vectors convey whether a super-pixel is on (retains the original input values) or off (set to the corresponding baseline value, e.g. black image). An interpretable linear model is trained with input being the binary vectors and outputs as the corresponding scores of the image classifier with the appropriate super-pixels masked based on the binary vector. Coefficients of the trained surrogate linear model convey the importance of each super-pixel.
