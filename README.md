# Text-classifier-using-Embedding-Layer
## AIM
To create a classifier using specialized layers for text data such as Embedding and GlobalAveragePooling1D.

## PROBLEM STATEMENT 
The program enables us to classify the given BBC dataset into its respective areas like different categories, for example buisness, sports and tech using Deep learning techniques, which includes loading and preprocessing the data, creating the neural network model, training and evaluation its performance.

## DESIGN STEPS
## STEP 1:
Unzip the zip file and load the BBC news dataset, split it into training and validation dataset.

## STEP 2:
Implement a function to convert the text into lower cases, remove the stop words and eliminate punctuation.


## STEP 3:
Create a TextVectorizer layer to tokenize and convert the dataset into sequences for model training.

##  STEP 4:
Use TensorFlow's StringLookup layer to encode text labels into numerical format.

## STEP 5:
Use TensorFlow's StringLookup layer to encode text labels into numerical format.

## STEP 6:
Train the model for 30 epochs using the prepared training data and validate its performance on the validation set.

## STEP 7:
Evaluate the model's accuracy and loss, and plot the results to track performance over time.


## PROGRAM

```python


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import zipfile

# Unzipping and reading the CSV file
with zipfile.ZipFile('/content/BBC News Train.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_data')

with open("extracted_data/BBC News Train.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"The second line (first data point) looks like this:\n\n{csvfile.readline()}")

# Defining useful global variables
VOCAB_SIZE = 1000  # Maximum number of words to keep
EMBEDDING_DIM = 16  # Dimension of the dense embedding
MAX_LENGTH = 120  # Maximum length of sequences
TRAINING_SPLIT = 0.8  # Proportion of data used for training

# Loading and pre-processing the data
data_dir = "extracted_data/BBC News Train.csv"
data = np.loadtxt(data_dir, delimiter=',', skiprows=1, dtype='str', comments=None)
print(f"Shape of the data: {data.shape}")
print(f"{data[0]}\n{data[1]}")

# Test the function
print(f"There are {len(data)} sentence-label pairs in the dataset.\n")
print(f"First sentence has {len((data[0, 1]).split())} words.\n")
print(f"The first 5 labels are {data[:5, 2]}")

# train_val_datasets function to split data into training and validation sets
def train_val_datasets(data, train_split=0.8):
    '''
    Splits data into training and validation sets

    Args:
        data (np.array): array with two columns, first one is the label, the second is the text

    Returns:
        (tf.data.Dataset, tf.data.Dataset): tuple containing the train and validation datasets
    '''
    # Compute the number of samples that will be used for training
    train_size = int(len(data) * train_split)

    # Slice the dataset to get only the texts and labels
    texts = data[:, 1]
    labels = data[:, 2]

    # Split the texts and labels into train/validation splits
    train_texts = texts[:train_size]
    validation_texts = texts[train_size:]
    train_labels = labels[:train_size]
    validation_labels = labels[train_size:]

    # Create the train and validation datasets from the splits
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_texts, validation_labels))

    return train_dataset, validation_dataset

# Create the datasets
train_dataset, validation_dataset = train_val_datasets(data)
print(f"There are {train_dataset.cardinality()} sentence-label pairs for training.\n")
print(f"There are {validation_dataset.cardinality()} sentence-label pairs for validation.\n")

# Define the text standardization function
def standardize_func(sentence):
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "more", "most", "my", "nor", "of", "on", "once", "only", "or", "out", "over", "same", "she", "so", "some", "that", "the", "their", "them", "there", "they", "this", "to", "up", "was", "we", "were", "what", "when", "where", "which", "who", "with", "you"]
    sentence = tf.strings.lower(sentence)
    for word in stopwords:
        sentence = tf.strings.regex_replace(sentence, rf"\b{word}\b", "")
    sentence = tf.strings.regex_replace(sentence, r'[!"#$%&()\*\+,-\./:;<=>?@^_`{|}~\']', "")
    return sentence

# fit_vectorizer function to create and adapt TextVectorization layer
def fit_vectorizer(train_sentences, standardize_func):
    if isinstance(train_sentences, np.ndarray):
        train_sentences = tf.data.Dataset.from_tensor_slices(train_sentences)

    vectorizer = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=MAX_LENGTH, standardize=standardize_func)
    vectorizer.adapt(train_sentences)

    return vectorizer

# Create the vectorizer
text_only_dataset = train_dataset.map(lambda text, label: text)
vectorizer = fit_vectorizer(text_only_dataset, standardize_func)
vocab_size = vectorizer.vocabulary_size()
print(f"Vocabulary contains {vocab_size} words\n")

# fit_label_encoder function for label encoding
def fit_label_encoder(train_labels, validation_labels):
    all_labels = train_labels.concatenate(validation_labels)
    label_encoder = tf.keras.layers.StringLookup(mask_token=None, num_oov_indices=0)
    label_encoder.adapt(all_labels)

    return label_encoder

# Create the label encoder
train_labels_only = train_dataset.map(lambda text, label: label)
validation_labels_only = validation_dataset.map(lambda text, label: label)
label_encoder = fit_label_encoder(train_labels_only, validation_labels_only)
print(f'Unique labels: {label_encoder.get_vocabulary()}')

# preprocess_dataset function
def preprocess_dataset(dataset, text_vectorizer, label_encoder):
    dataset = dataset.map(lambda text, label: (text_vectorizer(text), label_encoder(label)))
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

# Preprocess your dataset
train_proc_dataset = preprocess_dataset(train_dataset, vectorizer, label_encoder)
validation_proc_dataset = preprocess_dataset(validation_dataset, vectorizer, label_encoder)

# Model creation function
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # Assuming 5 categories
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Model evaluation on sample data
example_batch = train_proc_dataset.take(1)
try:
    model.evaluate(example_batch, verbose=False)
except:
    print("Your model is not compatible with the dataset you defined earlier.")
else:
    predictions = model.predict(example_batch, verbose=False)
    print(f"Predictions have shape: {predictions.shape}")

# Model training
history = model.fit(train_proc_dataset, epochs=30, validation_data=validation_proc_dataset)

# Plot function
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()

# Plot accuracy and loss
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```



## OUTPUT

![Screenshot 2024-10-21 114659](https://github.com/user-attachments/assets/bbebab65-ac6b-4376-bf24-fded683d74de)





### Loss, Accuracy Vs Iteration Plot
![download](https://github.com/user-attachments/assets/a189c8b1-9d2c-45f0-a7be-01e47cd17be4)


![download](https://github.com/user-attachments/assets/a80102ea-64f3-4b60-ae62-9d7cacd8dfeb)

## RESULT
Thus the program to create a classifier using specialized layers for text data such as Embedding and GlobalAveragePooling1D is implemented successfully.
