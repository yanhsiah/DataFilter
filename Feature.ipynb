{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pywt\n",
    "# (cA, cD) = pywt.dwt([1, 2, 3, 4, 5, 6], 'db1')\n",
    "# ([2.12132034 4.94974747 7.77817459], [-0.70710678 -0.70710678 -0.70710678])\n",
    "# https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import svm\n",
    "# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html\n",
    "from matplotlib import style\n",
    "# style.use('ggplot')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[4.95072450e-05 1.79903582e-04 3.89218330e-05 2.74255870e-05]\n",
      " [4.82313330e-05 1.79841183e-04 3.02605330e-05 2.27624550e-05]\n",
      " [4.26322220e-05 1.73451379e-04 3.33227220e-05 2.34134500e-05]\n",
      " [4.26322220e-05 1.73451379e-04 3.33227220e-05 2.34134500e-05]\n",
      " [4.20603900e-05 1.70729123e-04 3.85809690e-05 2.34134500e-05]\n",
      " [3.95718960e-05 1.65071338e-04 3.25217840e-05 2.44528060e-05]\n",
      " [3.95718960e-05 1.65071338e-04 3.25217840e-05 2.44528060e-05]\n",
      " [3.29511240e-05 1.60115771e-04 3.63439320e-05 2.44528060e-05]\n",
      " [3.19154930e-05 1.57691538e-04 3.45185400e-05 2.42860990e-05]\n",
      " [2.77431680e-05 1.52460299e-04 2.92453910e-05 1.32350250e-05]\n",
      " [2.40607190e-05 1.48248859e-04 2.64681880e-05 1.32350250e-05]\n",
      " [2.32327730e-05 1.45413913e-04 1.76522880e-05 3.86871400e-06]\n",
      " [2.16225160e-05 1.42143108e-04 1.85566020e-05 3.86871400e-06]\n",
      " [2.20388170e-05 1.42300501e-04 1.94925810e-05 4.70504200e-06]\n",
      " [2.26655970e-05 1.43103302e-04 1.81552020e-05 4.75347000e-06]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([ 6.0675650e-07,  1.9990840e-06,  0.0000000e+00,  0.0000000e+00,\n",
       "       -1.5846455e-06,  3.5492705e-06,  8.3260250e-07, -1.9110740e-06,\n",
       "       -5.2945700e-07, -2.8889625e-06, -1.0035000e-06, -2.7520550e-07,\n",
       "       -1.2945400e-07, -4.9825500e-08,  0.0000000e+00,  0.0000000e+00])"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"filtered_chocolate/0.csv\", usecols=['ch0', 'ch1', 'ch2', 'ch3'])\n",
    "print(df.loc[0:14].values)\n",
    "dwt(df.loc[0:14])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[5.2820891 4.7696754 4.9255788 4.5990571 4.2547472 4.0634535 3.1762756\n",
      " 2.2802502 2.4748035 2.3787841 2.1195039 1.7683953 1.9039027 1.9016676\n",
      " 1.7171726]\n",
      "<class 'numpy.ndarray'>\n",
      "[5.2820891 4.7696754 4.9255788 4.5990571 4.2547472 4.0634535 3.1762756\n",
      " 2.2802502 2.4748035 2.3787841 2.1195039 1.7683953 1.9039027 1.9016676\n",
      " 1.7171726 5.2820891 4.7696754 4.9255788 4.5990571 4.2547472 4.0634535\n",
      " 3.1762756 2.2802502 2.4748035 2.3787841 2.1195039 1.7683953 1.9039027\n",
      " 1.9016676 1.7171726]\n",
      "[ 5.2820891  4.7696754  4.9255788  4.5990571  4.2547472  4.0634535\n",
      "  3.1762756  2.2802502  2.4748035  2.3787841  2.1195039  1.7683953\n",
      "  1.9039027  1.9016676  1.7171726 18.5689889 18.1193464 18.1288458\n",
      " 17.9178081 17.1777792 16.4050609 15.7933682 15.1385553 15.0495209\n",
      " 14.8945488 14.3660232 14.42004   14.6069564 14.4448131 14.2258592\n",
      "  4.8656017  5.0429255  4.3747015  3.8227066  4.4032931  3.9415434\n",
      "  3.1800009  2.8452836  2.2577122  2.2732653  1.9093044  2.0206906\n",
      "  2.1368265  1.9024126  1.9686297  5.2035786  4.9107708  4.5713969\n",
      "  3.875792   4.2439438  3.6852434  2.7358532  1.7192215  2.0516105\n",
      "  1.8314458  1.2445264  0.8787028  0.6867573  0.8613802  1.0106713]\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv(\"filtered_pasta/0.csv\", usecols=['ch0', 'ch1', 'ch2', 'ch3'])\n",
    "arr = df.loc[0:14].div(0.00001)[\"ch0\"].values\n",
    "print(arr)\n",
    "print(type(arr))\n",
    "\n",
    "arr = np.append(arr, arr)\n",
    "print(arr)\n",
    "# dwt(df.loc[0:14].values)\n",
    "print(raw(df.loc[0:14]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [],
   "source": [
    "def wt(segment):\n",
    "    # featureSize = 8\n",
    "    return pywt.wavedec(segment[\"ch0\"], 'db1', level=3)[2]\n",
    "\n",
    "def dwt(segment):\n",
    "    # featureSize = 16\n",
    "    coeffs = pywt.dwt2(segment.values, 'haar')\n",
    "    cA, (cH, cV, cD) = coeffs\n",
    "    res = cA.flatten()\n",
    "    return res\n",
    "\n",
    "def raw(segment):\n",
    "    segment = segment.div(0.00001)\n",
    "    # featureSize = 4 * batchSize + 16\n",
    "    res = segment[\"ch0\"].values\n",
    "    res = np.append(res, segment[\"ch1\"].values)\n",
    "    res = np.append(res, segment[\"ch2\"].values)\n",
    "    res = np.append(res, segment[\"ch3\"].values)\n",
    "    res = np.append(res, dwt(segment))\n",
    "    return res\n",
    "\n",
    "featureSize = 76\n",
    "def getFeature(segment):\n",
    "    return raw(segment)\n",
    "\n",
    "def featuresFromFile(fileName, fileNum):\n",
    "    global featureSize\n",
    "    batch = 15\n",
    "    df = pd.read_csv(\"filtered_\" + fileName + \"/\" + str(fileNum) + \".csv\", usecols=['ch0', 'ch1', 'ch2', 'ch3'])\n",
    "    pickups = np.empty(shape=[0,featureSize])\n",
    "    putdowns = np.empty(shape=[0,featureSize])\n",
    "\n",
    "    rows = df.shape[0]\n",
    "    turns = rows / batch\n",
    "    turns -= turns % 2\n",
    "    rows = turns * batch\n",
    "    r = 0\n",
    "    while r < rows:\n",
    "        # pick-up\n",
    "        pickups = np.vstack((pickups, getFeature(df.loc[r:r+batch-1])))\n",
    "        r += batch\n",
    "        # put-down\n",
    "        putdowns = np.vstack((putdowns, getFeature(df.loc[r:r+batch-1])))\n",
    "        r += batch\n",
    "    return (pickups, putdowns)\n",
    "\n",
    "def labelFeatures(pickupFeatures, pickupLabels, putdownFeatures, putdownLabels, fileName, fileNum, label):\n",
    "    (pickups, putdowns) = featuresFromFile(fileName, fileNum)\n",
    "    # register features\n",
    "    pickupFeatures = np.vstack((pickupFeatures, pickups))\n",
    "    putdownFeatures = np.vstack((putdownFeatures, putdowns))\n",
    "    # register labels\n",
    "    pickupLabels = np.hstack((pickupLabels, np.full((1, pickups.shape[0]), label)))\n",
    "    putdownLabels = np.hstack((putdownLabels, np.full((1, putdowns.shape[0]), label + 1)))\n",
    "    return (pickupFeatures, pickupLabels, putdownFeatures, putdownLabels)\n",
    "\n",
    "def trainModels(features, labels):\n",
    "    # features\n",
    "    X = features\n",
    "    # labels\n",
    "    y = labels.flatten()\n",
    "    clf = svm.SVC(C=1, kernel='poly', coef0=1.0, probability=True, tol=1e-5)\n",
    "    clf.fit(X, y)\n",
    "    return clf\n",
    "\n",
    "def predict(pickupModel, putdownModel, fileName, fileRange, answerLabel):\n",
    "    pickupResult = np.array([], dtype=int)\n",
    "    putdownResult = np.array([], dtype=int)\n",
    "    pickupCorrect = 0\n",
    "    putdownCorrect = 0\n",
    "    for fileNum in fileRange:\n",
    "        (pickups, putdowns) = featuresFromFile(fileName, fileNum)\n",
    "        pickupResult = np.append(pickupResult, pickupModel.predict(pickups))\n",
    "        putdownResult = np.append(putdownResult, putdownModel.predict(putdowns))\n",
    "    for label in pickupResult:\n",
    "        if label == answerLabel:\n",
    "            pickupCorrect += 1\n",
    "    for label in putdownResult:\n",
    "            if label == (answerLabel + 1):\n",
    "                putdownCorrect += 1\n",
    "    print(pickupResult)\n",
    "    print(putdownResult)\n",
    "    print(\"{0:.2f}\".format(100.0 * pickupCorrect / pickupResult.size) + '%')\n",
    "    print(\"{0:.2f}\".format(100.0 * putdownCorrect / putdownResult.size) + '%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/allenh/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.\n",
      "  \"avoid this warning.\", FutureWarning)\n",
      "/Users/allenh/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.\n",
      "  \"avoid this warning.\", FutureWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 1 1 1 1 3 1]\n",
      "[2 4 2 2 2 4 2 4 2]\n",
      "88.89%\n",
      "66.67%\n",
      "[3 1 3 1 3 3 3 3 1 3 1]\n",
      "[4 2 2 2 2 2 2 6 4 6 2]\n",
      "63.64%\n",
      "18.18%\n",
      "[5 5 5 3 5 5 5 5 5 5 5]\n",
      "[6 6 6 6 6 6 6 6 6 6 6]\n",
      "90.91%\n",
      "100.00%\n"
     ]
    }
   ],
   "source": [
    "# 1: pick up chocolate\n",
    "# 2: put down chocolate\n",
    "# 3: pick up peach\n",
    "# 4: put down peach\n",
    "# 5: pick up pasta\n",
    "# 6: put down pasta\n",
    "\n",
    "totalFiles = 16\n",
    "predictNo = 15\n",
    "\n",
    "pickupFeatures = np.empty(shape=[0, featureSize])\n",
    "pickupLabels = np.empty(shape=[1, 0],dtype=int)\n",
    "putdownFeatures = np.empty(shape=[0, featureSize])\n",
    "putdownLabels = np.empty(shape=[1, 0],dtype=int)\n",
    "\n",
    "for fileNum in range(predictNo):\n",
    "    (pickupFeatures, pickupLabels, putdownFeatures, putdownLabels) = labelFeatures(pickupFeatures, pickupLabels, putdownFeatures, putdownLabels, \"chocolate\", fileNum, 1)\n",
    "    (pickupFeatures, pickupLabels, putdownFeatures, putdownLabels) = labelFeatures(pickupFeatures, pickupLabels, putdownFeatures, putdownLabels, \"peach\", fileNum, 3)\n",
    "    (pickupFeatures, pickupLabels, putdownFeatures, putdownLabels) = labelFeatures(pickupFeatures, pickupLabels, putdownFeatures, putdownLabels, \"pasta\", fileNum, 5)\n",
    "\n",
    "pickupModel = trainModels(pickupFeatures, pickupLabels)\n",
    "putdownModel = trainModels(putdownFeatures, putdownLabels)\n",
    "\n",
    "predict(pickupModel, putdownModel, \"chocolate\", range(predictNo, totalFiles), 1)\n",
    "predict(pickupModel, putdownModel, \"peach\", range(predictNo, totalFiles), 3)\n",
    "predict(pickupModel, putdownModel, \"pasta\", range(predictNo, totalFiles), 5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 124\n",
    "# 61.36%\n",
    "# 56.82%\n",
    "# 31.03%\n",
    "# 22.41%\n",
    "# 87.93%\n",
    "# 82.76%\n",
    "\n",
    "# 76\n",
    "# 61.36%\n",
    "# 56.82%\n",
    "# 31.03%\n",
    "# 25.86%\n",
    "# 87.93%\n",
    "# 84.48%\n",
    "\n",
    "# 60\n",
    "# 61.36%\n",
    "# 50.00%\n",
    "# 31.03%\n",
    "# 24.14%\n",
    "# 89.66%\n",
    "# 82.76%\n",
    "\n",
    "# 8/10\n",
    "# 72.22%\n",
    "# 55.56%\n",
    "# 77.27%\n",
    "# 68.18%\n",
    "# 86.36%\n",
    "# 86.36%\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
