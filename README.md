# Titanic-Competition
Kaggle titanic competation 
1	Import libraries
2	Import files
3	General view of data and data description
4	EDA
  •	Describe
  •	Missing values
5	Feature Analysis
6	Wanted column vs every other columns
  •	train[['Sex', 'Survived']].groupby('Sex', as_index = False).mean()
  •	sns.barplot(x = 'Sex', y ='Survived', data = train);
  •	sns.factorplot(x = 'Pclass', y = 'Survived', hue = 'Sex', data = train, kind = 'bar');
  •	detect outliers (writing functions, 4 types of detect outliers)
  •	sns.heatmap(train[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot = True, fmt = '.2f', cmap = 'coolwarm');
  •	sns.distplot(train['Age'],);
  •	sns.FacetGrid(train, col = 'Survived');
  •	map(sns.distplot, 'Age');
  •	sns.kdeplot(train['Age'][train['Survived'] == 1], label = 'Survived');
7	Data Processing
  •	train = train.drop(['Cabin'], axis = 1)
  •	Missing values in training set   train.isnull().sum().sort_values(ascending = False)
  •	Compute the most frequent value of Embarked in training set
mode = train['Embarked'].dropna().mode()[0]
  •	Compute median
median = test['Fare'].dropna().median()
  •	Fill missing value  with median
test['Fare'].fillna(median, inplace = True)
      o	sns.distplot(combine['Fare'], label = 'Skewness: %.2f'%(combine['Fare'].skew()))
plt.legend(loc = 'best');
      o	Apply log transformation to Fare column to reduce skewness
combine['Fare'] = combine['Fare'].map(lambda x: np.log(x) if x > 0 else 0)
  •	Like IsAlone, Title   deduction should be made






8	Modelling 
  •	Scikit-learn
    o	Split training data
        X_train = train.drop('Survived', axis = 1)
        Y_train = train['Survived']
        X_test = test.drop('PassengerId', axis = 1).copy()
        print("X_train shape: ", X_train.shape)
        print("Y_train shape: ", Y_train.shape)
        print("X_test shape: ", X_test.shape)
  •	Random forest (many more)
        random_forest = RandomForestClassifier(n_estimators = 100)
        random_forest.fit(X_train, Y_train)
        Y_pred = random_forest.predict(X_test)
        acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
        acc_random_forest
9	Training Accuracy
  •	models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
                                 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 
                                 'Linear SVC', 'Decision Tree'],
                       'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron,
                                 acc_sgd, acc_linear_svc, acc_decision_tree]})

  models.sort_values(by = 'Score', ascending = False, ignore_index = True)

  •	K- Fold Cross validation



10	Preparing data for submission
  •	submit.to_csv(r"C:\Users\udea3\OneDrive\Masaüstü\DS\aHaziran2022\titanic_results.csv", index = False)



