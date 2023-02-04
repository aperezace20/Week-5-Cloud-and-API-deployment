{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "o0FydRedUP27"
   },
   "source": [
    "# Turning a Jupyter notebook into a web app\n",
    "\n",
    "---\n",
    "\n",
    "### This notebook is designed to be used alongside Anvil's [turning a Jupyter notebook into a web app tutorial](https://anvil.works/learn/tutorials/jupyter-to-web-app).\n",
    "\n",
    "The text cells below tell you the steps you need to take to connect this notebook to an Anvil app. The steps are:\n",
    "\n",
    "\n",
    "1. Install the `anvil-uplink` library\n",
    "2. Import the `anvil.server` package\n",
    "3. Connect the notebook using your apps Uplink key\n",
    "4. Create a function to call from your app that includes the `anvil.server.callable` decorator\n",
    "5. Add `anvil.server.wait_forever()` to the end of the notebook\n",
    "\n",
    "### Follow along below for more detail.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "WOjHWnhO5k0x"
   },
   "source": [
    "### Let's start by importing the Anvil server package by adding `import anvil.server`:\n",
    "\n",
    "Importing `anvil.server` means, when this notebook is connected via the Uplink, it will behave in the same way as any other [Anvil Server Module](https://anvil.works/docs/server)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "EML6wBYQ5fiM",
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import anvil.server"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "RV2ze8a7ScHo"
   },
   "source": [
    "### Then connect this notebook to your app using your Uplink key `anvil.server.connect(\"your-uplink-key\")`:\n",
    "\n",
    "For information on how to get your apps Uplink key, see [Step 4 - Enable the Uplink](https://anvil.works/learn/tutorials/google-colab-to-web-app#step-4-enable-the-uplink)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "MA9-qSCOSckw"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Connecting to wss://anvil.works/uplink\n",
      "Anvil websocket open\n",
      "Connected to \"Default Environment\" as SERVER\n"
     ]
    }
   ],
   "source": [
    "import anvil.server\n",
    "\n",
    "anvil.server.connect(\"server_UTTW2EMQNB4WE2JYSQHM5AEE-Q3XSBZYLJSKIHZAV\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Mlf9a7vM_PVF"
   },
   "source": [
    "### Build and train the classification model\n",
    "\n",
    "The next cell gets the dataset, finds an optimal number of neighbours and then builds and trains the model. How this works is outside the scope of this tutorial, however, if you want to read more about how the code below works, Towards Data Science has a useful article [here](https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75).\n",
    "\n",
    "We don't need to change anything in the next cell.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 51
    },
    "colab_type": "code",
    "id": "Z9FGGe-2-V79",
    "outputId": "bdcc7644-1eef-4205-f1e4-490f6f78d52c"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(n_neighbors=10)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "from sklearn import metrics\n",
    "from sklearn.datasets import load_iris\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "iris = load_iris()\n",
    "X = iris.data\n",
    "y = iris.target\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=4)\n",
    "\n",
    "# The following code is used only when needing to find the optimal n_neighbors\n",
    "\"\"\"\n",
    "scores = {}\n",
    "scores_list = []\n",
    "k_range = range(1, 26)\n",
    "for k in k_range:\n",
    "  knn = KNeighborsClassifier(n_neighbors=k)\n",
    "  knn.fit(X_train, y_train)\n",
    "  y_pred = knn.predict(X_test)\n",
    "  scores[k] = metrics.accuracy_score(y_test, y_pred)\n",
    "  scores_list.append(metrics.accuracy_score(y_test, y_pred))\n",
    "\n",
    "plt.plot(k_range,scores_list) \n",
    "plt.xlabel('Value of K for KNN')\n",
    "plt.ylabel('Testing Accuracy')\n",
    "\"\"\"\n",
    "\n",
    "knn = KNeighborsClassifier(n_neighbors=10)\n",
    "knn.fit(X,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Gdil_W7b-N9Z"
   },
   "source": [
    "### Finally, we will create our `predict_iris()` function with a `@anvil.server.callable` decorator. The decorator makes the function callable from our Anvil app. \n",
    "Add the following code to the next cell:\n",
    "```\n",
    "@anvil.server.callable\n",
    "def predict_iris(sepal_length, sepal_width, petal_length, petal_width):\n",
    "  classification = knn.predict([[sepal_length, sepal_width, petal_length, petal_width]])\n",
    "  return iris.target_names[classification][0]\n",
    "```\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "ChnTYxx3-MRt"
   },
   "outputs": [],
   "source": [
    "@anvil.server.callable\n",
    "def predict_iris(sepal_length, sepal_width, petal_length, petal_width):\n",
    "  classification = knn.predict([[sepal_length, sepal_width, petal_length, petal_width]])\n",
    "  return iris.target_names[classification][0]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "WR1p147uXX0z"
   },
   "source": [
    "<!-- ---\n",
    "\n",
    "## That's it, 5 simple steps to connect your notebook to your Anvil app! \n",
    "\n",
    "--- -->"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "iris-classifier-start.ipynb",
   "provenance": []
  },
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
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
