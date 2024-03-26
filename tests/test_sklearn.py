# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=no-member
# pylint: disable=too-many-lines
# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=unused-import
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
"""sklearn estimators (de)serialization test suite"""


import numpy as np
import pandas as pd
from common import from_json, to_from_json, to_json
from numpy.testing import assert_array_equal
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.calibration import *
from sklearn.cluster import *
from sklearn.compose import *
from sklearn.covariance import *
from sklearn.cross_decomposition import *
from sklearn.datasets import *
from sklearn.decomposition import *
from sklearn.discriminant_analysis import *
from sklearn.ensemble import *
from sklearn.exceptions import *
from sklearn.feature_extraction import *
from sklearn.feature_selection import *
from sklearn.gaussian_process import *
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.impute import *
from sklearn.inspection import *
from sklearn.isotonic import *
from sklearn.kernel_approximation import *
from sklearn.kernel_ridge import *
from sklearn.linear_model import *
from sklearn.manifold import *
from sklearn.metrics import *
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.mixture import *
from sklearn.model_selection import *
from sklearn.multiclass import *
from sklearn.multioutput import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.random_projection import *
from sklearn.semi_supervised import *
from sklearn.svm import *
from sklearn.tree import *


def _to_json_and_back(obj: BaseEstimator) -> BaseEstimator:
    return from_json(to_json({"e": obj}))["e"]


def _fit_labels_test(
    obj: BaseEstimator, x: np.ndarray
) -> tuple[BaseEstimator, BaseEstimator]:
    obj = obj.fit(x)
    obj2 = _to_json_and_back(obj)
    assert_array_equal(obj.labels_, obj.labels_)
    return obj, obj2


def _fit_predict_x_z_test(
    obj: BaseEstimator, x: np.ndarray, z: np.ndarray
) -> tuple[BaseEstimator, BaseEstimator]:
    obj = obj.fit(x)
    obj2 = _to_json_and_back(obj)
    assert_array_equal(obj.predict(z), obj2.predict(z))
    return obj, obj2


def _fit_predict_x_y_z_test(
    obj: BaseEstimator, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[BaseEstimator, BaseEstimator]:
    obj = obj.fit(x, y)
    obj2 = _to_json_and_back(obj)
    assert_array_equal(obj.predict(z), obj2.predict(z))
    return obj, obj2


def _fit_score_test(
    obj: BaseEstimator,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> tuple[BaseEstimator, BaseEstimator]:
    obj = obj.fit(x_train, y_train)
    obj2 = _to_json_and_back(obj)
    assert_array_equal(obj.score(x_test, y_test), obj2.score(x_test, y_test))
    return obj, obj2


def _fit_transform_x_test(
    obj: BaseEstimator,
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[BaseEstimator, BaseEstimator]:
    obj = obj.fit(x_train)
    obj2 = _to_json_and_back(obj)
    assert_array_equal(obj.transform(x_test), obj2.transform(x_test))
    return obj, obj2


def _fit_transform_x_y_test(
    obj: BaseEstimator,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray | None = None,
) -> tuple[BaseEstimator, BaseEstimator]:
    obj = obj.fit(x_train, y_train)
    obj2 = _to_json_and_back(obj)
    if y_test is None:
        x_tr1 = obj.transform(x_test)
        x_tr2 = obj2.transform(x_test)
        assert_array_equal(x_tr1, x_tr2)
    else:
        x_tr1, y_tr1 = obj.transform(x_test, y_test)
        x_tr2, y_tr2 = obj2.transform(x_test, y_test)
        assert_array_equal(x_tr1, x_tr2)
        assert_array_equal(y_tr1, y_tr2)
    return obj, obj2


def _test_cross_decomposition_estimator(
    obj: BaseEstimator,
) -> tuple[BaseEstimator, BaseEstimator]:
    x = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2.0, 5.0, 4.0]]
    y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    obj = obj.fit(x, y)
    obj2 = _to_json_and_back(obj)
    x1, y1 = obj.transform(x, y)
    x2, y2 = obj2.transform(x, y)
    assert_array_equal(x1, x2)
    assert_array_equal(y1, y2)
    return obj, obj2


def _test_covariance_estimator(
    obj: BaseEstimator,
) -> tuple[BaseEstimator, BaseEstimator]:
    """Unit tests for covariance estimators"""
    np.random.seed(0)
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    kw = {"mean": [0, 0, 0, 0], "cov": true_cov, "size": 200}
    x = np.random.multivariate_normal(**kw)
    x_test = np.random.multivariate_normal(**kw)
    obj = obj.fit(x)
    obj2 = _to_json_and_back(obj)
    assert_array_equal(obj.score(x_test), obj2.score(x_test))
    assert_array_equal(obj.mahalanobis(x_test), obj2.mahalanobis(x_test))
    return obj, obj2


def test_affinitypropagation():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html
    """
    x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    y = [[0, 0], [4, 4]]
    _fit_predict_x_z_test(AffinityPropagation(random_state=0), x, y)


def test_agglomerativeclustering():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    """
    x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    _fit_labels_test(AgglomerativeClustering(), x)


# def test_birch():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
#     """
#     x = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
#     e = Birch(n_clusters=None)
#     e.fit(x)
#     e2 = _to_json_and_back(e)
#     assert_array_equal(e.predict(x), e2.predict(x))


def test_dbscan():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    """
    x = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    _fit_labels_test(DBSCAN(eps=3, min_samples=2), x)


def test_optics():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
    """
    x = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]])
    _fit_labels_test(OPTICS(min_samples=2), x)


def test_kmeans():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """
    x = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y = [[0, 0], [12, 3]]
    _fit_predict_x_z_test(
        KMeans(n_clusters=2, random_state=0, n_init="auto"), x, y
    )


# def test_bisectingkmeans():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html
#     """
#     x = np.array(
#         [
#             [1, 2],
#             [1, 4],
#             [1, 0],
#             [10, 2],
#             [10, 4],
#             [10, 0],
#             [10, 6],
#             [10, 8],
#             [10, 10],
#         ]
#     )
#     y = [[0, 0], [12, 3]]
#     e = BisectingKMeans(n_clusters=3, random_state=0).fit(x)
#     e2 = _to_json_and_back(e)
#     assert_array_equal(e.predict(y), e2.predict(y))


def test_minibatchkmeans():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
    """
    x = np.array(
        [
            [1, 2],
            [1, 4],
            [1, 0],
            [4, 2],
            [4, 0],
            [4, 4],
            [4, 5],
            [0, 1],
            [2, 2],
            [3, 2],
            [5, 5],
            [1, -1],
        ]
    )
    z = [[0, 0], [4, 4]]
    e = MiniBatchKMeans(
        n_clusters=2, random_state=0, batch_size=6, max_iter=10, n_init="auto"
    )
    _fit_predict_x_z_test(e, x, z)


def test_meanshift():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    """
    x = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    y = [[0, 0], [5, 5]]
    _fit_predict_x_z_test(MeanShift(bandwidth=2), x, y)


def test_spectralclustering():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
    """
    x = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    e = SpectralClustering(
        n_clusters=2, assign_labels="discretize", random_state=0
    )
    _fit_labels_test(e, x)


def test_spectralbiclustering():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralBiclustering.html
    """
    x = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    e = SpectralBiclustering(n_clusters=2, random_state=0).fit(x)
    e2 = _to_json_and_back(e)
    assert_array_equal(e.row_labels_, e2.row_labels_)
    assert_array_equal(e.column_labels_, e2.column_labels_)


def test_spectralcoclustering():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralCoclustering.html
    """
    x = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
    e = SpectralCoclustering(n_clusters=2, random_state=0).fit(x)
    e2 = _to_json_and_back(e)
    assert_array_equal(e.row_labels_, e2.row_labels_)
    assert_array_equal(e.column_labels_, e2.column_labels_)


def test_ellipticenvelope():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html
    """
    np.random.seed(0)
    true_cov = np.array([[0.8, 0.3], [0.3, 0.4]])
    x = np.random.multivariate_normal(mean=[0, 0], cov=true_cov, size=500)
    y = [[0, 0], [3, 3]]
    _fit_predict_x_z_test(EllipticEnvelope(random_state=0), x, y)


def test_empiricalcovariance():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html
    """
    _test_covariance_estimator(EmpiricalCovariance())


def test_graphicallasso():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html
    """
    _test_covariance_estimator(GraphicalLasso())


def test_ledoitwolf():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
    """
    _test_covariance_estimator(LedoitWolf())


def test_mincovdet():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
    """
    _test_covariance_estimator(MinCovDet(random_state=0))


def test_oas():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html
    """
    _test_covariance_estimator(OAS())


def test_shrunkcovariance():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html
    """
    _test_covariance_estimator(ShrunkCovariance())


def test_plscanonical():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSCanonical.html
    """
    _test_cross_decomposition_estimator(PLSCanonical(n_components=2))


def test_plsregression():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
    """
    _test_cross_decomposition_estimator(PLSRegression(n_components=2))


def test_plssvd():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html
    """
    _test_cross_decomposition_estimator(PLSSVD(n_components=2))


def test_cca():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html
    """
    _test_cross_decomposition_estimator(CCA(n_components=1))


def test_dictionarylearning():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html
    """
    x1, _, _ = make_sparse_coded_signal(
        n_samples=100,
        n_components=15,
        n_features=20,
        n_nonzero_coefs=10,
        random_state=42,
    )
    x2, _, _ = make_sparse_coded_signal(
        n_samples=100,
        n_components=15,
        n_features=20,
        n_nonzero_coefs=10,
        random_state=43,
    )
    e = DictionaryLearning(
        n_components=15,
        transform_algorithm="lasso_lars",
        transform_alpha=0.1,
        random_state=42,
        max_iter=10,
    )
    _fit_transform_x_test(e, x1, x2)


# def test_fastica():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
#     """
#     x, _ = load_digits(return_X_y=True)
#     e = FastICA(n_components=7, random_state=0, whiten="unit-variance")
#     _fit_transform_x_test(e, x, x)


def test_incrementalpca():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html
    """
    x, _ = load_digits(return_X_y=True)
    y = sparse.csr_matrix(x)
    e = IncrementalPCA(n_components=7, batch_size=200)
    e.partial_fit(x[:100, :])
    e2 = _to_json_and_back(e)
    assert_array_equal(e.fit_transform(y), e2.fit_transform(y))


def test_kernelpca():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
    """
    x, _ = load_digits(return_X_y=True)
    e = KernelPCA(n_components=7, kernel="linear")
    _fit_transform_x_test(e, x, x)


def test_minibatchdictionarylearning():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchDictionaryLearning.html
    """
    x1, _, _ = make_sparse_coded_signal(
        n_samples=30,
        n_components=15,
        n_features=20,
        n_nonzero_coefs=10,
        random_state=42,
    )
    x2, _, _ = make_sparse_coded_signal(
        n_samples=30,
        n_components=15,
        n_features=20,
        n_nonzero_coefs=10,
        random_state=43,
    )
    dict_learner = MiniBatchDictionaryLearning(
        n_components=15,
        batch_size=3,
        transform_algorithm="lasso_lars",
        transform_alpha=0.1,
        max_iter=20,
        random_state=42,
    )
    _fit_transform_x_test(dict_learner, x1, x2)


# def test_minibatchnmf():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html
#     """
#     x, _, _ = make_sparse_coded_signal(
#         n_samples=100,
#         n_components=15,
#         n_features=20,
#         n_nonzero_coefs=10,
#         random_state=42,
#     )
#     e = MiniBatchDictionaryLearning(
#         n_components=15,
#         batch_size=3,
#         transform_algorithm="lasso_lars",
#         transform_alpha=0.1,
#         random_state=42,
#     )
#     _fit_transform_x_test(e, x, x)


def test_minibatchsparsepca():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html
    """
    x, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
    e = MiniBatchSparsePCA(
        n_components=5, batch_size=50, max_iter=10, random_state=0
    )
    e.fit(x)
    _fit_transform_x_test(e, x, x)


def test_nmf():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    """
    x = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    e = NMF(n_components=2, init="random", random_state=0)
    _fit_transform_x_test(e, x, x)


def test_pca():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    e = PCA(n_components=2)
    e, e2 = _fit_transform_x_test(e, x, x)
    assert_array_equal(e.score(x), e2.score(x))


def test_sparsepca():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html
    """
    x, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
    _fit_transform_x_test(SparsePCA(n_components=5, random_state=0), x, x)


def test_factoranalysis():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html
    """
    x, _ = load_digits(return_X_y=True)
    x = x[:10]
    e = FactorAnalysis(n_components=7, random_state=0, max_iter=10)
    _fit_transform_x_test(e, x, x)


def test_truncatedsvd():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    """
    np.random.seed(0)
    x_dense = np.random.rand(100, 100)
    x_dense[:, 2 * np.arange(50)] = 0
    x = sparse.csr_matrix(x_dense)
    e = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    _fit_transform_x_test(e, x, x)


def test_latentdirichletallocation():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    """
    x, _ = make_multilabel_classification(random_state=0)
    e = LatentDirichletAllocation(n_components=5, random_state=0)
    _fit_transform_x_test(e, x, x[-2:])


def test_randomforestclassifier():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    x, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    z = [[0, 0, 0, 0]]
    e = RandomForestClassifier(max_depth=2, random_state=0)
    e.fit(x, y)
    e2 = _to_json_and_back(e)
    assert_array_equal(e.predict(z), e2.predict(z))


def test_randomforestregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """
    x, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    z = [[0, 0, 0, 0]]
    e = RandomForestRegressor(max_depth=2, random_state=0)
    e.fit(x, y)
    e2 = _to_json_and_back(e)
    assert_array_equal(e.predict(z), e2.predict(z))


# def test_randomtreesembedding():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html
#     """
#     x = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
#     e = RandomTreesEmbedding(n_estimators=5, random_state=0, max_depth=1)
#     _fit_transform_x_test(e, x, x)


def test_extratreesclassifier():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    """
    x, y = make_classification(n_features=4, random_state=0)
    z = [[0, 0, 0, 0]]
    e = ExtraTreesClassifier(n_estimators=100, random_state=0)
    _fit_predict_x_y_z_test(e, x, y, z)


def test_extratreesregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
    """
    x, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    e = ExtraTreesRegressor(n_estimators=100, random_state=0)
    _fit_score_test(e, x_train, x_test, y_train, y_test)


# def test_baggingclassifier():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html
#     """
#     x, y = make_classification(
#         n_samples=100,
#         n_features=4,
#         n_informative=2,
#         n_redundant=0,
#         random_state=0,
#         shuffle=False,
#     )
#     z = [[0, 0, 0, 0]]
#     e = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0)
#     _fit_predict_x_y_z_test(e, x, y, z)


# def test_baggingregressor():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
#     """
#     x, y = make_classification(
#         n_samples=100,
#         n_features=4,
#         n_informative=2,
#         n_redundant=0,
#         random_state=0,
#         shuffle=False,
#     )
#     z = [[0, 0, 0, 0]]
#     e = BaggingClassifier(estimator=SVR(), n_estimators=10, random_state=0)
#     _fit_predict_x_y_z_test(e, x, y, z)


def test_isolationforest():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """
    x = [[-1.1], [0.3], [0.5], [100]]
    _fit_predict_x_z_test(IsolationForest(random_state=0), x, x)


# def test_gradientboostingclassifier():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
#     """
#     x, y = make_hastie_10_2(random_state=0)
#     x_train, x_test = x[:2000], x[2000:]
#     y_train, y_test = y[:2000], y[2000:]
#     e = GradientBoostingClassifier(
#         n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
#     )
#     _fit_score_test(e, x_train, x_test, y_train, y_test)


# def test_gradientboostingregressor():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
#     """
#     x, y = make_regression(random_state=0)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
#     e = GradientBoostingRegressor(random_state=0)
#     e, e2 = _fit_predict_x_y_z_test(e, x_train, y_train, x_test[1:2])
#     assert_array_equal(e.score(x_test, y_test), e2.score(x_test, y_test))


def test_adaboostclassifier():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    """
    x, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    z = [[0, 0, 0, 0]]
    e = AdaBoostClassifier(n_estimators=100, random_state=0)
    e, e2 = _fit_predict_x_y_z_test(e, x, y, z)
    assert_array_equal(e.score(x, y), e2.score(x, y))


def test_adaboostregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
    """
    x, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    z = [[0, 0, 0, 0]]
    e = AdaBoostRegressor(random_state=0, n_estimators=100)
    e, e2 = _fit_predict_x_y_z_test(e, x, y, z)
    assert_array_equal(e.score(x, y), e2.score(x, y))


# def test_histgradientboostingclassifier():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
#     """
#     x, y = load_iris(return_X_y=True)
#     e = HistGradientBoostingClassifier(max_iter=10)
#     _fit_score_test(e, x, x, y, y)


# def test_histgradientboostingregressor():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
#     """
#     x, y = load_diabetes(return_X_y=True)
#     _fit_score_test(HistGradientBoostingRegressor(), x, x, y, y)


def test_variancethreshold():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
    """
    x = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
    _fit_transform_x_test(VarianceThreshold(), x, x)


# def test_gaussianprocessregressor():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
#     """
#     x, y = make_friedman2(n_samples=500, noise=0, random_state=0)
#     kernel = DotProduct() + WhiteKernel()
#     e = GaussianProcessRegressor(kernel=kernel, random_state=0)
#     e, e2 = _fit_score_test(e, x, x, y, y)
#     assert_array_equal(
#         e.predict(x[:2, :], return_std=True),
#         e2.predict(x[:2, :], return_std=True),
#     )
#     assert_array_equal(
#         e.predict(x[:2, :], return_cov=True),
#         e2.predict(x[:2, :], return_cov=True),
#     )


# def test_gaussianprocessclassifier():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
#     """
#     x, y = load_iris(return_X_y=True)
#     kernel = 1.0 * RBF(1.0)
#     e = GaussianProcessClassifier(
#         kernel=kernel, random_state=0, max_iter_predict=10
#     )
#     e, e2 = _fit_score_test(e, x, x, y, y)
#     assert_array_equal(
#         e.predict_proba(x[:2, :]),
#         e2.predict_proba(x[:2, :]),
#     )


# def test_isotonicregression():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
#     """
#     x, y = make_regression(n_samples=10, n_features=1, random_state=41)
#     z = [.1, .2]
#     _fit_predict_x_y_z_test(IsotonicRegression(), x, y, z)


def test_ardregression():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html
    """
    x, y, z = [[0, 0], [1, 1], [2, 2]], [0, 1, 2], [[1, 1]]
    _fit_predict_x_y_z_test(ARDRegression(max_iter=10), x, y, z)


def test_bayesianridge():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
    """
    x, y, z = [[0, 0], [1, 1], [2, 2]], [0, 1, 2], [[1, 1]]
    _fit_predict_x_y_z_test(BayesianRidge(), x, y, z)


def test_elasticnet():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    """
    x, y = make_regression(n_features=2, random_state=0)
    z = [[0, 0]]
    _fit_predict_x_y_z_test(ElasticNet(random_state=0, max_iter=10), x, y, z)


def test_huberregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html
    """
    rng = np.random.RandomState(0)
    x, y, _ = make_regression(
        n_samples=200, n_features=2, noise=4.0, coef=True, random_state=0
    )
    x[:4] = rng.uniform(10, 20, (4, 2))
    y[:4] = rng.uniform(10, 20, 4)
    e = HuberRegressor()
    e, e2 = _fit_score_test(e, x, x, y, y)
    assert_array_equal(
        e.predict(x[:1, :]),
        e2.predict(x[:1, :]),
    )


def test_lars():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html
    """
    x, y = [[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111]
    _fit_predict_x_y_z_test(Lars(n_nonzero_coefs=1), x, y, x)


def test_lasso():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """
    x, y = [[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111]
    _fit_predict_x_y_z_test(Lasso(alpha=0), x, y, x)


def test_lassolars():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html
    """
    x, y = [[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111]
    _fit_predict_x_y_z_test(LassoLars(alpha=0.01), x, y, x)


def test_lassolarsic():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html
    """
    x = [[-2, 2], [-1, 1], [0, 0], [1, 1], [2, 2]]
    y = [-2.2222, -1.1111, 0, -1.1111, -2.2222]
    _fit_predict_x_y_z_test(LassoLarsIC(criterion="bic"), x, y, x)


def test_linearregression():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """
    x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(x, np.array([1, 2])) + 3
    z = np.array([[3, 5]])
    e, e2 = _fit_score_test(LinearRegression(), x, x, y, y)
    assert_array_equal(e.predict(z), e2.predict(z))


def test_logisticregression():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    x, y = load_iris(return_X_y=True)
    e = LogisticRegression(random_state=0, max_iter=100).fit(x, y)
    e2 = _to_json_and_back(e)
    assert_array_equal(e.score(x, y), e2.score(x, y))


def test_multitaskelasticnet():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html
    """
    x, y = [[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]]
    _fit_score_test(MultiTaskElasticNet(alpha=0.1), x, x, y, y)


def test_multitasklasso():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html
    """
    x, y = [[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]]
    _fit_score_test(MultiTaskLasso(alpha=0.1), x, x, y, y)


def test_orthogonalmatchingpursuit():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
    """
    x, y = make_regression(noise=4, random_state=0)
    e, e2 = _fit_score_test(OrthogonalMatchingPursuit(), x, x, y, y)
    assert_array_equal(e.predict(x[:1, :]), e2.predict(x[:1, :]))


# def test_passiveaggressiveclassifier():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
#     """
#     x, y = make_regression(noise=4, random_state=0)
#     e = PassiveAggressiveClassifier(max_iter=10, random_state=0)
#     e, e2 = _fit_score_test(e, x, x, y, y)
#     assert_array_equal(e.predict(x), e2.predict(x))


def test_passiveaggressiveregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html
    """
    x, y = make_regression(noise=4, random_state=0)
    e = PassiveAggressiveRegressor(max_iter=10, random_state=0)
    e, e2 = _fit_score_test(e, x, x, y, y)
    assert_array_equal(e.predict(x), e2.predict(x))


# def test_perceptron():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
#     """
#     x, y = load_digits(return_X_y=True)
#     _fit_score_test(Perceptron(tol=1e-3, random_state=0), x, x, y, y)


def test_quantileregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html
    """
    n_samples, n_features = 10, 2
    rng = np.random.RandomState(0)
    x, y = rng.randn(n_samples, n_features), rng.randn(n_samples)
    e = QuantileRegressor(quantile=0.8, solver="highs")
    _fit_score_test(e, x, x, y, y)


def test_ridge():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    """
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    x, y = rng.randn(n_samples, n_features), rng.randn(n_samples)
    _fit_score_test(Ridge(alpha=1.0), x, x, y, y)


def test_ridgeclassifier():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html
    """
    x, y = load_breast_cancer(return_X_y=True)
    _fit_score_test(RidgeClassifier(), x, x, y, y)


# def test_sgdclassifier():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
#     """
#     x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
#     y = np.array([1, 1, 2, 2])
#     x = StandardScaler().fit_transform(x)
#     _fit_score_test(SGDClassifier(max_iter=10, tol=1e-3), x, x, y, y)


def test_sgdregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html
    """
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    x, y = rng.randn(n_samples, n_features), rng.randn(n_samples)
    x = StandardScaler().fit_transform(x)
    _fit_score_test(SGDRegressor(max_iter=10, tol=1e-3), x, x, y, y)


# def test_sgdoneclasssvm():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDOneClassSVM.html
#     """
#     x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
#     _fit_predict_x_z_test(SGDOneClassSVM(random_state=42), x, x)


def test_theilsenregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html
    """
    x, y = make_regression(
        n_samples=200, n_features=2, noise=4.0, random_state=0
    )
    _fit_score_test(TheilSenRegressor(random_state=0), x, x, y, y)


def test_ransacregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html
    """
    x, y = make_regression(
        n_samples=200, n_features=2, noise=4.0, random_state=0
    )
    _fit_score_test(RANSACRegressor(random_state=0), x, x, y, y)


# def test_poissonregressor():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
#     """
#     x, y = [[1, 2], [2, 3], [3, 4], [4, 3]], [12, 17, 22, 21]
#     _fit_score_test(PoissonRegressor(), x, x, y, y)


# def test_gammaregressor():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html
#     """
#     x, y = [[1, 2], [2, 3], [3, 4], [4, 3]], [19, 26, 33, 30]
#     _fit_score_test(GammaRegressor(), x, x, y, y)


# def test_tweedieregressor():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html
#     """
#     x, y = [[1, 2], [2, 3], [3, 4], [4, 3]], [2, 3.5, 5, 5.5]
#     _fit_score_test(TweedieRegressor(), x, x, y, y)


def test_locallylinearembedding():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html
    """
    x, _ = load_digits(return_X_y=True)
    e = LocallyLinearEmbedding(n_components=2)
    xe = e.fit_transform(x[:100])
    e2 = _to_json_and_back(e)
    assert_array_equal(xe, e2.embedding_)


def test_isomap():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html
    """
    x, _ = load_digits(return_X_y=True)
    e = Isomap(n_components=2)
    xe = e.fit_transform(x[:100])
    e2 = _to_json_and_back(e)
    assert_array_equal(xe, e2.embedding_)


def test_mds():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    """
    x, _ = load_digits(return_X_y=True)
    e = MDS(n_components=2, normalized_stress="auto")
    xe = e.fit_transform(x[:100])
    e2 = _to_json_and_back(e)
    assert_array_equal(xe, e2.embedding_)


def test_spectralembedding():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html
    """
    x, _ = load_digits(return_X_y=True)
    e = SpectralEmbedding(n_components=2)
    xe = e.fit_transform(x[:100])
    e2 = _to_json_and_back(e)
    assert_array_equal(xe, e2.embedding_)


def test_tsne():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    """
    x = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    e = TSNE(n_components=2, perplexity=3, n_iter=250)
    xe = e.fit_transform(x)
    e2 = _to_json_and_back(e)
    assert_array_equal(xe, e2.embedding_)


def test_gaussianmixture():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    """
    x = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    gm = GaussianMixture(n_components=2, random_state=0).fit(x)
    gm2 = _to_json_and_back(gm)
    y = [[0, 0], [12, 3]]
    assert_array_equal(gm.predict(y), gm2.predict(y))
    assert_array_equal(gm.score(x), gm2.score(x))


def test_bayesiangaussianmixture():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html
    """
    x = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    gm = BayesianGaussianMixture(n_components=2, random_state=0).fit(x)
    gm2 = _to_json_and_back(gm)
    y = [[0, 0], [12, 3]]
    assert_array_equal(gm.predict(y), gm2.predict(y))
    assert_array_equal(gm.score(x), gm2.score(x))


def test_bernoullinb():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
    """
    rng = np.random.RandomState(1)
    x, y = rng.randint(5, size=(6, 100)), np.array([1, 2, 3, 4, 4, 5])
    _fit_predict_x_y_z_test(BernoulliNB(force_alpha=True), x, y, x[2:3])


def test_gaussiannb():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    """
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    z = [[-0.8, -1]]
    _fit_predict_x_y_z_test(GaussianNB(), x, y, z)


def test_multinomialnb():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    """
    rng = np.random.RandomState(1)
    x, y = rng.randint(5, size=(6, 100)), np.array([1, 2, 3, 4, 5, 6])
    _fit_predict_x_y_z_test(MultinomialNB(force_alpha=True), x, y, x[2:3])


def test_complementnb():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html
    """
    rng = np.random.RandomState(1)
    x, y = rng.randint(5, size=(6, 100)), np.array([1, 2, 3, 4, 5, 6])
    _fit_predict_x_y_z_test(ComplementNB(force_alpha=True), x, y, x[2:3])


def test_categoricalnb():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html
    """
    rng = np.random.RandomState(1)
    x, y = rng.randint(5, size=(6, 100)), np.array([1, 2, 3, 4, 5, 6])
    _fit_predict_x_y_z_test(CategoricalNB(force_alpha=True), x, y, x[2:3])


def test_kneighborsclassifier():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """
    x, y, z = [[0], [1], [2], [3]], [0, 0, 1, 1], [[1.1]]
    _fit_predict_x_y_z_test(KNeighborsClassifier(n_neighbors=3), x, y, z)


def test_kneighborsregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    """
    x, y, z = [[0], [1], [2], [3]], [0, 0, 1, 1], [[1.1]]
    _fit_predict_x_y_z_test(KNeighborsRegressor(n_neighbors=3), x, y, z)


# def test_kneighborstransformer():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsTransformer.html
#     """
#     x, _ = load_wine(return_X_y=True)
#     e = KNeighborsTransformer(n_neighbors=5, mode="distance")
#     _fit_transform_x_test(e, x, x)


def test_nearestcentroid():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
    """
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    z = [[-0.8, -1]]
    _fit_predict_x_y_z_test(NearestCentroid(), x, y, z)


def test_nearestneighbors():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """
    x, z = [[0, 0, 2], [1, 0, 0], [0, 0, 1]], [[0, 0, 1.3]]
    e = NearestNeighbors(n_neighbors=2, radius=0.4)
    e.fit(x)
    e2 = _to_json_and_back(e)
    assert_array_equal(
        e.kneighbors(z, 2, return_distance=False),
        e2.kneighbors(z, 2, return_distance=False),
    )


def test_radiusneighborsclassifier():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html
    """
    x, y, z = [[0], [1], [2], [3]], [0, 0, 1, 1], [[1.5]]
    _fit_predict_x_y_z_test(RadiusNeighborsClassifier(radius=1.0), x, y, z)


def test_radiusneighborsregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html
    """
    x, y, z = [[0], [1], [2], [3]], [0, 0, 1, 1], [[1.5]]
    _fit_predict_x_y_z_test(RadiusNeighborsRegressor(radius=1.0), x, y, z)


# def test_radiusneighborstransformer():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsTransformer.html
#     """
#     x, _ = load_wine(return_X_y=True)
#     e = RadiusNeighborsTransformer(radius=42.0, mode="distance")
#     _fit_transform_x_test(e, x, x)


def test_kerneldensity():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html
    """
    rng = np.random.RandomState(42)
    x = rng.random_sample((100, 3))
    e = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(x)
    e2 = _to_json_and_back(e)
    assert_array_equal(e.score_samples(x[:3]), e2.score_samples(x[:3]))


def test_localoutlierfactor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
    """
    x = [[-1.1], [0.2], [101.1], [0.3]]
    e = LocalOutlierFactor(n_neighbors=2, novelty=True)
    _fit_predict_x_z_test(e, x, x)


def test_neighborhoodcomponentsanalysis():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html
    """
    x, y = load_iris(return_X_y=True)
    e1 = NeighborhoodComponentsAnalysis(random_state=42)
    x_tr1 = e1.fit_transform(x, y)
    e2 = _to_json_and_back(e1)
    x_tr2 = e2.transform(x)
    assert_array_equal(x_tr1, x_tr2)


def test_bernoullirbm():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html
    """
    x = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    _fit_transform_x_test(BernoulliRBM(n_components=2), x, x)


# def test_mlpclassifier():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#     """
#     x, y = make_classification(n_samples=100, random_state=1)
#     x_train, x_test, y_train, y_test = train_test_split(
#         x, y, stratify=y, random_state=1
#     )
#     e = MLPClassifier(random_state=1, max_iter=10)
#     _fit_score_test(e, x_train, x_test, y_train, y_test)


# def test_mlpregressor():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
#     """
#     x, y = make_classification(n_samples=100, random_state=1)
#     x_train, x_test, y_train, y_test = train_test_split(
#         x, y, stratify=y, random_state=1
#     )
#     e = MLPRegressor(random_state=1, max_iter=10)
#     _fit_score_test(e, x_train, x_test, y_train, y_test)


def test_binarizer():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html
    """
    x = [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
    _fit_transform_x_test(Binarizer(), x, x)


# def test_kbinsdiscretizer():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
#     """
#     x = [[-2, 1, -4, -1], [-1, 2, -3, -0.5], [0, 3, -2, 0.5], [1, 4, -1, 2]]
#     e = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
#     _fit_transform_x_test(e, x, x)


def test_kernelcenterer():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KernelCenterer.html
    """
    x = [[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]]
    k = pairwise_kernels(x, metric="linear")
    _fit_transform_x_test(KernelCenterer(), k, k)


def test_labelbinarizer():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
    """
    x, z = [1, 2, 6, 4, 2], [1, 6]
    _fit_transform_x_test(LabelBinarizer(), x, z)


def test_labelencoder():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    """
    x = [1, 2, 2, 6]
    _fit_transform_x_test(LabelEncoder(), x, x)


def test_multilabelbinarizer():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
    """
    x = [(1, 2), (3,)]
    _fit_transform_x_test(MultiLabelBinarizer(), x, x)


def test_minmaxscaler():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    x = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    _fit_transform_x_test(MinMaxScaler(), x, x)


def test_maxabsscaler():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html
    """
    x = [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
    _fit_transform_x_test(MaxAbsScaler(), x, x)


def test_quantiletransformer():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
    """
    rng = np.random.RandomState(0)
    x = np.sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)
    e = QuantileTransformer(n_quantiles=10, random_state=0)
    _fit_transform_x_test(e, x, x)


def test_normalizer():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
    """
    x = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
    _fit_transform_x_test(Normalizer(), x, x)


def test_powertransformer():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    """
    x = [[1, 2], [3, 2], [4, 5]]
    _fit_transform_x_test(PowerTransformer(), x, x)


def test_robustscaler():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    """
    x = [[1.0, -2.0, 2.0], [-2.0, 1.0, 3.0], [4.0, 1.0, -2.0]]
    _fit_transform_x_test(RobustScaler(), x, x)


# def test_splinetransformer():
#     """
#     https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html
#     """
#     x = np.arange(6).reshape(6, 1)
#     _fit_transform_x_test(SplineTransformer(degree=2, n_knots=3), x, x)


def test_standardscaler():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """
    x = [[0, 0], [0, 0], [1, 1], [1, 1]]
    _fit_transform_x_test(StandardScaler(), x, x)


def test_polynomialfeatures():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    """
    x = np.arange(6).reshape(3, 2)
    _fit_transform_x_test(PolynomialFeatures(2), x, x)


def test_sparserandomprojection():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html
    """
    rng = np.random.RandomState(42)
    x = rng.rand(25, 3000)
    _fit_transform_x_test(SparseRandomProjection(random_state=rng), x, x)


def test_gaussianrandomprojection():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html
    """
    rng = np.random.RandomState(42)
    x = rng.rand(25, 3000)
    _fit_transform_x_test(GaussianRandomProjection(random_state=rng), x, x)


def test_labelpropagation():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html
    """
    iris = load_iris()
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    labels = np.copy(iris.target)
    labels[random_unlabeled_points] = -1
    e = LabelPropagation()
    _fit_score_test(e, iris.data, iris.data, labels, labels)


def test_labelspreading():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html
    """
    iris = load_iris()
    rng = np.random.RandomState(42)
    random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
    labels = np.copy(iris.target)
    labels[random_unlabeled_points] = -1
    e = LabelSpreading()
    _fit_score_test(e, iris.data, iris.data, labels, labels)


def test_linearsvc():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    """
    x, y = make_classification(n_features=4, random_state=0)
    e = LinearSVC(random_state=0, tol=1e-5, dual=True)
    _fit_score_test(e, x, x, y, y)


def test_linearsvr():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
    """
    x, y = make_classification(n_features=4, random_state=0)
    e = LinearSVR(random_state=0, tol=1e-5, dual=True)
    _fit_score_test(e, x, x, y, y)


def test_nusvc():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html
    """
    x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    _fit_score_test(NuSVC(), x, x, y, y)


def test_nusvr():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html
    """
    x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    _fit_score_test(NuSVR(), x, x, y, y)


def test_oneclasssvm():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
    """
    x = [[0], [0.44], [0.45], [0.46], [1]]
    _fit_predict_x_z_test(OneClassSVM(gamma="auto"), x, x)


def test_svc():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    z = [[-0.8, -1]]
    _fit_predict_x_y_z_test(SVC(gamma="auto"), x, y, z)


def test_svr():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    """
    x = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])
    z = [[-0.8, -1]]
    _fit_predict_x_y_z_test(SVR(gamma="auto"), x, y, z)


def test_decisiontreeclassifier():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """
    iris = load_iris()
    x, y = iris.data, iris.target  # pylint: disable=no-member
    e = DecisionTreeClassifier(random_state=0)
    _fit_score_test(e, x, x, y, y)


def test_decisiontreeregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """
    x, y = load_diabetes(return_X_y=True)
    e = DecisionTreeRegressor(random_state=0)
    _fit_score_test(e, x, x, y, y)


def test_extratreeclassifier():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html
    """
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    e = ExtraTreeClassifier(random_state=0)
    _fit_score_test(e, x_train, x_test, y_train, y_test)


def test_extratreeregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html
    """
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    e = ExtraTreeRegressor(random_state=0)
    _fit_score_test(e, x_train, x_test, y_train, y_test)


def test_lineardiscriminantanalysis():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
    """
    x = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    y, z = [1, 1, 1, 2, 2, 2], [[-0.8, -1]]
    _fit_predict_x_y_z_test(LinearDiscriminantAnalysis(), x, y, z)


def test_quadraticdiscriminantanalysis():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
    """
    x = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    y, z = [1, 1, 1, 2, 2, 2], [[-0.8, -1]]
    _fit_predict_x_y_z_test(QuadraticDiscriminantAnalysis(), x, y, z)


def test_missingindicator():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html
    """
    x_train = np.array([[np.nan, 1, 3], [4, 0, np.nan], [8, 1, 0]])
    x_test = np.array([[5, 1, np.nan], [np.nan, 2, 3], [2, 4, 0]])
    _fit_transform_x_test(MissingIndicator(), x_train, x_test)


def test_simpleimputer():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    """
    x_train = np.array([[np.nan, 1, 3], [4, 0, np.nan], [8, 1, 0]])
    x_test = np.array([[5, 1, np.nan], [np.nan, 2, 3], [2, 4, 0]])
    _fit_transform_x_test(SimpleImputer(), x_train, x_test)


def test_knnimputer():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
    """
    x = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
    _fit_transform_x_test(KNNImputer(n_neighbors=2), x, x)


def test_polynomialcountsketch():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.PolynomialCountSketch.html
    """
    x = [[0, 0], [1, 1], [1, 0], [0, 1]]
    e = PolynomialCountSketch(degree=3, random_state=1)
    _fit_transform_x_test(e, x, x)


def test_rbfsampler():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html
    """
    x = [[0, 0], [1, 1], [1, 0], [0, 1]]
    e = RBFSampler(gamma=1, random_state=1)
    _fit_transform_x_test(e, x, x)


def test_skewedchi2sampler():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.SkewedChi2Sampler.html
    """
    x = [[0, 0], [1, 1], [1, 0], [0, 1]]
    e = SkewedChi2Sampler(skewedness=0.01, n_components=10, random_state=0)
    _fit_transform_x_test(e, x, x)


def test_additivechi2sampler():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler.html
    """
    x, y = load_digits(return_X_y=True)
    e = AdditiveChi2Sampler(sample_steps=2)
    e.fit(x, y)
    e2 = _to_json_and_back(e)
    assert_array_equal(e.transform(x), e2.transform(x))


def test_nystroem():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html
    """
    x, _ = load_digits(n_class=9, return_X_y=True)
    e = Nystroem(gamma=0.2, random_state=1, n_components=300)
    _fit_transform_x_test(e, x, x)


def test_kernelridge():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
    """
    n_samples, n_features = 10, 5
    rng = np.random.RandomState(0)
    x, y = rng.randn(n_samples, n_features), rng.randn(n_samples)
    _fit_score_test(KernelRidge(alpha=1.0), x, x, y, y)


def test_votingregressor():
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html
    """
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    # r3 = KNeighborsRegressor()
    x = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    y = np.array([2, 6, 12, 20, 30, 42])
    # er = VotingRegressor([("lr", r1), ("rf", r2), ("r3", r3)])
    er = VotingRegressor([("lr", r1), ("rf", r2)])
    _fit_predict_x_y_z_test(er, x, y, x)


def test_classifierchain():
    """https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html"""
    x, y = make_multilabel_classification(
        n_samples=12, n_classes=3, random_state=0
    )
    x_train, x_test, y_train, _ = train_test_split(x, y, random_state=0)
    base_lr = LogisticRegression(solver="lbfgs", random_state=0)
    chain = ClassifierChain(base_lr, order="random", random_state=0)
    _fit_predict_x_y_z_test(chain, x_train, y_train, x_test)


# def test_columntransformer_1():
#     """https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html"""
#     ct = ColumnTransformer(
#         [
#             ("norm1", Normalizer(norm="l1"), [0, 1]),
#             ("norm2", Normalizer(norm="l1"), slice(2, 4)),
#         ]
#     )
#     x = np.array([[0.0, 1.0, 2.0, 2.0], [1.0, 1.0, 0.0, 1.0]])
#     _fit_transform_x_test(ct, x, x)


# def test_columntransformer_2():
#     """https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html"""
#     x = pd.DataFrame(
#         {
#             "documents": [
#                 "First item",
#                 "second one here",
#                 "Is this the last?",
#             ],
#             "width": [3, 4, 5],
#         }
#     )
#     # "documents" is a string which configures ColumnTransformer to
#     # pass the documents column as a 1d array to the CountVectorizer
#     ct = ColumnTransformer(
#         [
#             ("text_preprocess", CountVectorizer(), "documents"),
#             ("num_preprocess", MinMaxScaler(), ["width"]),
#         ]
#     )
#     _fit_transform_x_test(ct, x, x)


def test_featureunion():
    """https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html"""
    union = FeatureUnion(
        [("pca", PCA(n_components=1)), ("svd", TruncatedSVD(n_components=2))]
    )
    x = [[0.0, 1.0, 3], [2.0, 2.0, 5]]
    _fit_transform_x_test(union, x, x)
    _fit_transform_x_test(union.set_params(svd__n_components=1), x, x)


def test_multioutputclassifier():
    """https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html"""
    x, y = make_multilabel_classification(n_classes=3, random_state=0)
    clf = MultiOutputClassifier(LogisticRegression())
    _fit_predict_x_y_z_test(clf, x, y, x[-2:])


def test_multioutputregressor():
    """https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html"""
    x, y = load_linnerud(return_X_y=True)
    regr = MultiOutputRegressor(Ridge(random_state=123))
    _fit_predict_x_y_z_test(regr, x, y, x[[0]])


def test_onevsoneclassifier():
    """https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html"""
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, _ = train_test_split(
        x, y, test_size=0.33, shuffle=True, random_state=0
    )
    clf = OneVsOneClassifier(LinearSVC(dual="auto", random_state=0))
    _fit_predict_x_y_z_test(clf, x_train, y_train, x_test[:10])


def test_onevsrestclassifier():
    """https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html"""
    x = np.array(
        [[10, 10], [8, 10], [-5, 5.5], [-5.4, 5.5], [-20, -20], [-15, -20]]
    )
    y = np.array([0, 0, 1, 1, 2, 2])
    clf = OneVsRestClassifier(LinearSVC(dual="auto", random_state=0))
    _fit_predict_x_y_z_test(clf, x, y, [[-19, -20], [9, 9], [-5, 5]])


# def test_outputcodeclassifier():
#     """https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OutputCodeClassifier.html"""
#     x, y = make_classification(
#         n_samples=100,
#         n_features=4,
#         n_informative=2,
#         n_redundant=0,
#         random_state=0,
#         shuffle=False,
#     )
#     clf = OutputCodeClassifier(
#         estimator=RandomForestClassifier(random_state=0), random_state=0
#     )
#     _fit_predict_x_y_z_test(clf, x, y, [[0, 0, 0, 0]])


def test_pipeline():
    """https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html"""
    x, y = make_classification(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(dual="auto", random_state=0)),
        ]
    )
    _fit_score_test(pipe, x_train, x_test, y_train, y_test)
    _fit_score_test(
        pipe.set_params(svc__C=10), x_train, x_test, y_train, y_test
    )


def test_regressorchain():
    """https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html"""
    logreg = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    x, y = [[1, 0], [0, 1], [1, 1]], [[0, 2], [1, 1], [2, 0]]
    chain = RegressorChain(base_estimator=logreg, order=[0, 1])
    _fit_predict_x_y_z_test(chain, x, y, x)


def test_rfe():
    """https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html"""
    x, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel="linear")
    s1 = RFE(estimator, n_features_to_select=5, step=1)
    s1 = s1.fit(x, y)
    s2 = to_from_json(s1)
    assert (s1.support_ == s2.support_).all()
    assert (s1.ranking_ == s2.ranking_).all()


# def test_rfecv():
#     """https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html"""
#     x, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
#     estimator = SVR(kernel="linear")
#     s1 = RFECV(estimator, step=1, cv=5)
#     s1 = s1.fit(x, y)
#     s2 = to_from_json(s1)
#     assert (s1.support_ == s2.support_).all()
#     assert (s1.ranking_ == s2.ranking_).all()


def test_selectfrommodel():
    """https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html"""
    x = [
        [0.87, -1.34, 0.31],
        [-2.79, -0.02, -0.85],
        [-1.34, -0.48, -2.55],
        [1.92, 1.48, 0.65],
    ]
    y = [0, 1, 0, 1]
    selector = SelectFromModel(estimator=LogisticRegression())
    _fit_transform_x_y_test(selector, x, x, y)


def test_selftrainingclassifier():
    """https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html"""
    rng = np.random.RandomState(42)
    iris = load_iris()
    random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.3
    iris.target[random_unlabeled_points] = -1
    m = SelfTrainingClassifier(RandomForestClassifier())
    _fit_predict_x_y_z_test(m, iris.data, iris.target, iris.data)


# def test_sequentialfeatureselector():
#     """https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html"""
#     x, y = load_iris(return_X_y=True)
#     clf = KNeighborsClassifier(n_neighbors=3)
#     sfs = SequentialFeatureSelector(clf, n_features_to_select=3)
#     _fit_transform_x_y_test(sfs, x, y, x)


def test_stackingclassifier():
    """https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html"""
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, random_state=42
    )
    estimators = [
        ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
        (
            "svr",
            make_pipeline(
                StandardScaler(), LinearSVC(dual="auto", random_state=42)
            ),
        ),
    ]
    clf = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )
    _fit_score_test(clf, x_train, x_test, y_train, y_test)


def test_stackingregressor():
    """https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html"""
    x, y = load_diabetes(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    estimators = [
        ("lr", RidgeCV()),
        ("svr", LinearSVR(dual="auto", random_state=42)),
    ]
    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor(
            n_estimators=10, random_state=42
        ),
    )
    _fit_score_test(reg, x_train, x_test, y_train, y_test)


def test_votingclassifier():
    """https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html"""
    clf1 = LogisticRegression(multi_class="multinomial", random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    eclf1 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="hard"
    )
    _fit_predict_x_y_z_test(eclf1, x, y, x)
    eclf2 = VotingClassifier(
        estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft"
    )
    _fit_predict_x_y_z_test(eclf2, x, y, x)
