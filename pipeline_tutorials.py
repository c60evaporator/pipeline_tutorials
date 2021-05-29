# データの読込
import seaborn as sns
iris = sns.load_dataset("iris")  # irisデータセットを使用
features = ['petal_width', 'petal_length', 'sepal_width']  # 説明変数
X = iris[features].values  # 説明変数をndarray化
y = iris['species']  # 目的変数をndarray化

# %% 学習データ=推論データのとき（パイプライン不使用）
from sklearn.pipeline import Pipeline  # パイプライン用クラス
from sklearn.preprocessing import StandardScaler  # 標準化クラス
from sklearn.decomposition import PCA  # 主成分分析クラス
from sklearn.svm import SVC  # サポートベクターマシン分類クラス
scaler = StandardScaler()
pca = PCA(n_components=2)
svm = SVC()
scaler.fit(X)  # 標準化の学習
X_scaler = scaler.transform(X)  # 標準化の推論
pca.fit(X_scaler)  # 主成分分析の学習
X_pca = pca.transform(X_scaler)  # 主成分分析の推論
svm.fit(X_pca, y)  # SVMの学習
pred = svm.predict(X_pca)  # SVMの推論
# %% 学習データ=推論データのとき（Pipelineクラス使用）
from sklearn.pipeline import Pipeline  # パイプライン用クラス
from sklearn.preprocessing import StandardScaler  # 標準化クラス
from sklearn.decomposition import PCA  # 主成分分析クラス
from sklearn.svm import SVC  # サポートベクターマシン分類クラス
pipe = Pipeline([("scaler", StandardScaler()),
                  ("pca", PCA(n_components=2)),
                  ("svm", SVC())])
pipe.fit(X, y)  # 学習
pred = pipe.predict(X)  # 推論

# %% 学習データと推論データを分けたとき（パイプライン不使用）
from sklearn.model_selection import train_test_split  # データ分割用メソッド
scaler = StandardScaler()
pca = PCA(n_components=2)
svm = SVC()
# 学習データと推論データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y)
# 学習データに対して学習実行
scaler.fit(X_train)  # 標準化の学習
X_train_scaler = scaler.transform(X)  # 標準化の推論
pca.fit(X_train_scaler)  # 主成分分析の学習
X_train_pca = pca.transform(X_train_scaler)  # 主成分分析の推論
svm.fit(X_train_pca, y)  # SVMの学習
# 推論データに対して推論実行
X_test_scaler = scaler.transform(X)  # 標準化の推論
X_test_pca = pca.transform(X_test_scaler)  # 主成分分析の推論
pred = svm.predict(X_test_pca)  # SVMの推論
# %% 学習データと推論データを分けたとき（Pipelineクラス使用）
from sklearn.model_selection import train_test_split  # データ分割用メソッド
pipe = Pipeline([("scaler", StandardScaler()),
                  ("pca", PCA(n_components=2)),
                  ("svm", SVC())])
# 学習データと推論データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y)
# 学習データに対して学習実行
pipe.fit(X_train, y_train)
# 推論データに対して推論実行
pred = pipe.predict(X_test)

# %% クロスバリデーションでスコア算出（パイプライン不使用）
from sklearn.model_selection import KFold  # クロスバリデーション分割用クラス
from sklearn.metrics import accuracy_score  # スコア(accuracy)算出用クラス
import numpy as np
scaler = StandardScaler()
pca = PCA(n_components=2)
svm = SVC()
# スコア保持用のリスト
scores = []
# クロスバリデーション実行
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # クロスバリデーション用クラス
for train, test in cv.split(X, y):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    # 学習データに対して学習実行
    scaler.fit(X_train)  # 標準化の学習
    X_train_scaler = scaler.transform(X)  # 標準化の推論
    pca.fit(X_train_scaler)  # 主成分分析の学習
    X_train_pca = pca.transform(X_train_scaler)  # 主成分分析の推論
    svm.fit(X_train_pca, y)  # SVMの学習
    # 推論データに対して推論実行
    X_test_scaler = scaler.transform(X)  # 標準化の推論
    X_test_pca = pca.transform(X_test_scaler)  # 主成分分析の推論
    pred = svm.predict(X_test_pca)  # SVMの推論
    # スコア算出
    accuracy = accuracy_score(y, pred)
    scores.append(accuracy)
# スコアを平均して最終指標とする
score = np.mean(scores) 

# %% 学習データと推論データを分けたとき（Pipelineクラス使用）
from sklearn.model_selection import KFold, cross_val_score  # クロスバリデーション指標算出用クラス
import numpy as np
pipe = Pipeline([("scaler", StandardScaler()),
                  ("pca", PCA(n_components=2)),
                  ("svm", SVC())])
# クロスバリデーション実行
cv = KFold(n_splits=5, shuffle=True, random_state=42)  # クロスバリデーション用クラス
scores = cross_val_score(pipe, X, y, scoring='accuracy')
# スコアを平均して最終指標とする
score = np.mean(scores) 
# %%
