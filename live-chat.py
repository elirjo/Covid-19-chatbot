{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "99af712b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CHATBOT IS LIVE, ASK AWAY\n",
      "Hello\n",
      "How are you feeling today?\n",
      "I am here to check if I have the virus\n",
      "How are you feeling today?\n",
      "headache, flu, vomit\n",
      "One or more of these have been listed in our symptoms list, are you vaccinated?\n",
      "Yes\n",
      "I have no name\n",
      "I have received vaccination\n",
      "Kindly check yourself for covid and get treated for covid.\n",
      "What is covid\n",
      "it is as deadly as a flu\n",
      "Can I go to campus\n",
      "How are you feeling today?\n",
      "I would like to enter campus\n",
      "Have you traveled out of the country in the past 14 days?\n",
      "no i have not\n",
      "Kindly check yourself for covid and get treated for covid.\n",
      "I think i'm sick\n",
      "Please provide your symptoms?\n",
      "can you screen me\n",
      "Have you traveled out of the country in the past 14 days?\n",
      "can you screen me\n",
      "Are you presently feeling some of the Symptoms?\n",
      "can you screen me\n",
      "Are you presently feeling some of the Symptoms?\n"
     ]
    }
   ],
   "source": [
    "#source https://www.youtube.com/watch?v=1lwddP0KUEg&t=318s\n",
    "\n",
    "import random\n",
    "import json\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "import nltk\n",
    "\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "\n",
    "intents = json.loads(open('intents.json').read())\n",
    "# yes_intents = json.loads(open('yes_intents.json').read())\n",
    "# no_intents = json.loads(open('no_intents.json').read())\n",
    "\n",
    "words = pickle.load(open('words.pkl', 'rb'))\n",
    "classes = pickle.load(open('classes.pkl', 'rb'))\n",
    "model = load_model('model.h5')\n",
    "\n",
    "def clean_up_sentence(sentence):\n",
    "    sentence_words = nltk.word_tokenize(sentence)\n",
    "    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]\n",
    "    return sentence_words\n",
    "\n",
    "def bag_of_words(sentence):\n",
    "    sentence_words = clean_up_sentence(sentence)\n",
    "    bag = [0] * len(words)\n",
    "    for w in sentence_words:\n",
    "        for i, word in enumerate(words) :\n",
    "            if word == w:\n",
    "                bag[i] = 1\n",
    "    return np.array(bag)\n",
    "\n",
    "def predict_class(sentence):\n",
    "    bow = bag_of_words(sentence)\n",
    "    res = model.predict(np.array([bow]))[0]\n",
    "    ERROR_THRESHOLD = 0.25\n",
    "    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]\n",
    "    results.sort(key = lambda x: x[1], reverse=True)\n",
    "    return_list = []\n",
    "    for r in results:\n",
    "        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})\n",
    "    return return_list\n",
    "\n",
    "def get_response(intents_list, intents_json):\n",
    "    tag = intents_list[0]['intent']\n",
    "    list_of_intents = intents_json['intents']\n",
    "    for i in list_of_intents:\n",
    "        if i['tag'] == tag:\n",
    "            result = random.choice(i['responses'])\n",
    "            break\n",
    "    return result\n",
    "\n",
    "print(\"CHATBOT IS LIVE, ASK AWAY\")\n",
    "\n",
    "while True:\n",
    "    message = input(\"\")\n",
    "    ints = predict_class(message)\n",
    "    res = get_response(ints, intents)\n",
    "    print(res)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10269ca0",
   "metadata": {},
   "outputs": [],
   "source": [
    "#synonmyns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea55e405",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#deploy and send link"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
