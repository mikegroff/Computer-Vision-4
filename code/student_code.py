import cv2
import numpy as np
import os.path as osp
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from IPython.core.debugger import set_trace
from utils import *
from random import shuffle


def get_tiny_images(image_paths):
  """
  This feature is inspired by the simple tiny images used as features in
  80 million tiny images: a large dataset for non-parametric object and
  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

  To build a tiny image feature, simply resize the original image to a very
  small square resolution, e.g. 16x16. You can either resize the images to
  square while ignoring their aspect ratio or you can crop the center
  square portion out of each image. Making the tiny images zero mean and
  unit length (normalizing them) will increase performance modestly.

  Useful functions:
  -   cv2.resize
  -   use load_image(path) to load a RGB images and load_image_gray(path) to
      load grayscale images

  Args:
  -   image_paths: list of N elements containing image paths

  Returns:
  -   feats: N x d numpy array of resized and then vectorized tiny images
            e.g. if the images are resized to 16x16, d would be 256
  """
  # dummy feats variable
  m = 16
  M = m**2
  v = np.ones((1,M))
  for i in image_paths:
      im = load_image_gray(i)
      im = cv2.resize(im, (m,m))
      im = im.reshape((1,M))
      im = im/np.linalg.norm(im)
      v = np.append(v, im , axis = 0)

  feats = v[1:]


  return feats

def build_vocabulary(image_paths, vocab_size):
  """
  This function will sample SIFT descriptors from the training images,
  cluster them with kmeans, and then return the cluster centers.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
          -  frames is a N x 2 matrix of locations, which can be thrown away
          here (but possibly used for extra credit in get_bags_of_sifts if
          you're making a "spatial pyramid").
          -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
  -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
          http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
               the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
               cluster_centers is a K x d matrix of cluster centers. This is
               your vocabulary.

  Args:
  -   image_paths: list of image paths.
  -   vocab_size: size of vocabulary

  Returns:
  -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
  """
  # Load images from the training set. To save computation time, you don't
  # necessarily need to sample from all images, although it would be better
  # to do so. You can randomly sample the descriptors from each image to save
  # memory and speed up the clustering. Or you can simply call vl_dsift with
  # a large step size here, but a smaller step size in get_bags_of_sifts.
  #
  # For each loaded image, get some SIFT features. You don't have to get as
  # many SIFT features as you will in get_bags_of_sift, because you're only
  # trying to get a representative sample here.
  #
  # Once you have tens of thousands of SIFT features from many training
  # images, cluster them with kmeans. The resulting centroids are now your
  # visual word vocabulary.

  dim = 128      # length of the SIFT descriptors that you are going to compute.

  des = np.ones((1,dim))
  for path in image_paths:
      img = load_image_gray(path)
      frames, descriptors = vlfeat.sift.dsift(img, step = 100, size = 50, fast=True,float_descriptors=True)
      des = np.vstack((des,descriptors))

  des = des[1:,:]
  vocab = vlfeat.kmeans.kmeans(des, vocab_size)

  return vocab

def get_bags_of_sifts(image_paths, vocab_filename):
  """
  This feature representation is described in the handout, lecture
  materials, and Szeliski chapter 14.
  You will want to construct SIFT features here in the same way you
  did in build_vocabulary() (except for possibly changing the sampling
  rate) and then assign each local feature to its nearest cluster center
  and build a histogram indicating how many times each cluster was used.
  Don't forget to normalize the histogram, or else a larger image with more
  SIFT features will look very different from a smaller version of the same
  image.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
          (but possibly used for extra credit in get_bags_of_sifts if you're
          making a "spatial pyramid").
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
  -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """
  # load vocabulary

  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)



  k,d = np.shape(vocab)
  feat = np.ones((1,k))
  vocab = np.float32(vocab)


  for path in image_paths:
      img = load_image_gray(path)
      frames, des = vlfeat.sift.dsift(img, step = 3, size = 5, fast=True,float_descriptors=True)
      asgns = vlfeat.kmeans.kmeans_quantize(des, vocab)
      f,b = np.histogram(asgns,bins = range(k+1))
      f = f/np.linalg.norm(f)
      feat = np.vstack((feat,f))
  # dummy features variable
  feats = feat[1:,:]


  return feats

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean'):
  """
  This function will predict the category for every test image by finding
  the training image with most similar features. Instead of 1 nearest
  neighbor, you can vote based on k nearest neighbors which will increase
  performance (although you need to pick a reasonable value for k).

  Useful functions:
  -   D = sklearn_pairwise.pairwise_distances(X, Y)
        computes the distance matrix D between all pairs of rows in X and Y.
          -  X is a N x d numpy array of d-dimensional features arranged along
          N rows
          -  Y is a M x d numpy array of d-dimensional features arranged along
          N rows
          -  D is a N x M numpy array where d(i, j) is the distance between row
          i of X and row j of Y

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating
          the ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  -   metric: (optional) metric to be used for nearest neighbor.
          Can be used to select different distance functions. The default
          metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
          well for histograms

  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  M,d = np.shape(test_image_feats)
  N,d = np.shape(train_image_feats)
  D = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats)
  label = []

  for row in D:
      ind = np.argmin(row)
      label.append(train_labels[ind])

  test_labels = label

  #############################################################################

  return  test_labels

def svm_classify(train_image_feats, train_labels, test_image_feats, extra = False):
  """
  This function will train a linear SVM for every category (i.e. one vs all)
  and then use the learned linear classifiers to predict the category of
  every test image. Every test feature will be evaluated with all 15 SVMs
  and the most confident SVM will "win". Confidence, or distance from the
  margin, is W*X + B where '*' is the inner product or dot product and W and
  B are the learned hyperplane parameters.

  Useful functions:
  -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  -   svm.fit(X, y)
  -   set(l)

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  test_labels = []
  N,d = np.shape(train_image_feats)
  # categories
  categories = list(set(train_labels))

  def classy(str,x):
      if x==str:
          return 1
      else:
          return -1


  # construct 1 vs all SVMs for each category

 


  if(extra == True):
      svms = {cat: SVC(tol=1e-5,max_iter = -1, C=150, gamma = 'scale',kernel = 'rbf', probability = True) for cat in categories}

      for s in svms:
          trainset = list(map(lambda x: classy(s,x), train_labels))
          svms[s].fit(train_image_feats,trainset)

      for f in test_image_feats:
          f = f.reshape((1,d))
          con = 0
          id = 0
          for s in svms:
              c = svms[s].predict_proba(f)[0,1]
              if (c > con):
                  id = 1
                  con = c
                  label = s
          test_labels.append(label)
  else:
      svms = {cat: LinearSVC(random_state=0, tol=1e-4, loss='hinge',max_iter = 10000, C=25) for cat in categories}

      for s in svms:
          trainset = list(map(lambda x: classy(s,x), train_labels))
          svms[s].fit(train_image_feats,trainset)

      for f in test_image_feats:
          con = -10000000
          for s in svms:
              W = svms[s].coef_
              B = svms[s].intercept_
              c = np.dot(W,f) + B
              if c > con:
                  con = c
                  label = s
          test_labels.append(label)


  #print(N,n)
  return test_labels

def l_result(pred,exp):
    n = len(pred)
    c = 0
    for i in range(0,n):
        if pred[i] == exp[i]:
            c+=1
    return c/n

def exp_results(trials, extra= False):
    num_train_per_cat = 100
    categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial', 'Suburb',
              'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry', 'Coast',
              'Mountain', 'Forest'];
    vsizes = [1000]
    avg_perm = []
    perf_std = []
    data_path = osp.join('..', 'data')
    train_image_paths, test_image_paths, train_labels, test_labels = get_image_paths(data_path,categories,num_train_per_cat)

    vocab_filename = 'vocab.pkl'
    for v in vsizes:
        print(v)
        tscore = []
        for i in range(0,trials):

            indices=np.arange(1500)
            shuffle(indices)
            train_paths = indices[0:1399]
            test_paths = indices[1400:-1]
            train_l = np.take(train_labels,train_paths)
            test_l = np.take(train_labels,test_paths)
            train_paths = np.take(train_image_paths,train_paths)
            test_paths = np.take(train_image_paths,test_paths)

            vocab_size = v  # Larger values will work better (to a point) but be slower to compute
            vocab = build_vocabulary(train_paths, vocab_size)
            with open(vocab_filename, 'wb') as f:
                pickle.dump(vocab, f)
                print('{:s} saved'.format(vocab_filename))

            train_image_feats = get_bags_of_sifts(train_paths, vocab_filename)
            test_feats = get_bags_of_sifts(test_paths, vocab_filename)
            pred = svm_classify(train_image_feats, train_l, test_feats, extra = extra)
            score = l_result(pred,test_l)
            tscore.append(score)
        tscore = np.asarray(tscore)
        avg_perm.append(np.mean(tscore))
        perf_std.append(np.std(tscore))


    return avg_perm, perf_std
