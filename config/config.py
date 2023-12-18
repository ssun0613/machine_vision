def get_config_dict():
    dataset_info = dict(
        name='wdm',
        train_path='./course_data/train_aug/',
        val_path='./course_data/test_aug/',
        height=52,
        width=52,
        channel=3,
        batch_size=10,
        num_workers=0,
    )

    path = dict(
        save_base_path='runs'
    )

    model = dict(
        name='resnet' # [ lenet | alexnet | resnet ]
    )

    solver = dict(
        name='sgd',
        gpu_id=0,
        lr0=1e-4,
        momentum=0.937,
        weight_decay=5e-4,
        max_epoch=50,
    )

    scheduler = dict(
        name='steplr'
    )

    # Merge all information into a dictionary variable
    config = dict(
        dataset=dataset_info,
        path=path,
        model=model,
        solver=solver,
        scheduler=scheduler,
    )

    return config
