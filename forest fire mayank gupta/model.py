from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

def get_models():
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            random_state=42
        )
    }
    return models
