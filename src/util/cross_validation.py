from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def kfold_iter(
        X, y, n_splits,
        random_state,
        is_campaign_group,
        campaign_ids=None,
        training_type='classification'):

    if is_campaign_group and campaign_ids is not None:
        return GroupKFold(n_splits=n_splits).split(X, y, campaign_ids)

    if training_type == 'classification':
        return StratifiedKFold(n_splits=n_splits,
                               random_state=random_state,
                               shuffle=True).split(X, y)
    elif training_type in ('regression', 'multi_regression'):
        return KFold(n_splits=n_splits,
                     random_state=random_state,
                     shuffle=True).split(X, y)
    else:
        raise ValueError('Invalid training type: {}'.format(training_type))
