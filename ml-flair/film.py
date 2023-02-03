def set_film_resnet18(model):
    # freeze all layers
    for layer in model.layers[:]:
        layer.trainable = False
    # enable head
    model.get_layer('classifier').trainable = True
    # enable batch norm weights (FilM)
    model.get_layer('initial_norm').trainable = True
    model.get_layer('layer1_block1_norm1').trainable = True
    model.get_layer('layer1_block1_norm2').trainable = True
    model.get_layer('layer1_block2_norm1').trainable = True
    model.get_layer('layer1_block2_norm2').trainable = True
    model.get_layer('layer2_block1_norm1').trainable = True
    model.get_layer('layer2_block1_norm2').trainable = True
    model.get_layer('layer2_block2_norm1').trainable = True
    model.get_layer('layer2_block2_norm2').trainable = True
    model.get_layer('layer3_block1_norm1').trainable = True
    model.get_layer('layer3_block1_norm2').trainable = True
    model.get_layer('layer3_block2_norm1').trainable = True
    model.get_layer('layer3_block2_norm2').trainable = True
    model.get_layer('layer4_block1_norm1').trainable = True
    model.get_layer('layer4_block1_norm2').trainable = True
    model.get_layer('layer4_block2_norm1').trainable = True
    model.get_layer('layer4_block2_norm2').trainable = True

