from chainer.training import triggers


def setup_record_trigger(training_type):

    if training_type == 'regression' or training_type == 'multi_regression':
        return triggers.MinValueTrigger('validation/main/loss')

    else:
        raise ValueError('Invalid training type: {}'.format(training_type))


def setup_optim_trigger(training_type):

    if training_type == 'regression' or training_type == 'multi_regression':
        return triggers.MinValueTrigger('validation/main/loss')

    else:
        raise ValueError('Invalid training type: {}'.format(training_type))


def setup_print_report_entries(training_type):

    if training_type == 'regression':
        entries = [
            'epoch',
            'main/loss', 'validation/main/loss',
            'elapsed_time', 'lr',
        ]
    elif training_type == 'multi_regression':
        entries = [
            'epoch',
            'main/loss', 'validation/main/loss',
            'main/loss_click', 'validation/main/loss_click',
            'main/loss_cv', 'validation/main/loss_cv',
            'elapsed_time', 'lr',
        ]
    else:
        raise ValueError('Invalid training type: {}'.format(training_type))

    return entries


def setup_plot_report_loss_entries(training_type):

    if training_type == 'classification' or training_type == 'regression':
        entries = ['main/loss', 'val/main/loss']

    elif training_type == 'multi_regression':
        entries = [
            'main/loss', 'validation/main/loss',
            'main/loss_click', 'validation/main/loss_click',
            'main/loss_cv', 'validation/main/loss_cv',
        ]
    else:
        raise ValueError('Invalid training type: {}'.format(training_type))

    return entries
