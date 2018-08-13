from __future__ import absolute_import, division, print_function
from keras import layers
from keras import objectives
from keras.engine import Input
from keras.engine import Model
from keras.regularizers import l2
from src import settings
from src.wavenet.causal_dilated_conv_1d import CausalDilatedConv1D
from keras.optimizers import Adam


def load():
    model = get()
    model.load_weights('models/wavenet_epoch-02_loss-5.6868_val_loss-5.6631.h5')
    return model

def get():
    model = _build_model()
    loss = objectives.categorical_crossentropy
    optim = _make_optimizer()

    model.compile(optimizer=optim, loss=loss)

    return model

def _make_optimizer():
    lr = 0.001
    momentum = 0.9
    decay = 0.0
    nesterov = True

    return Adam(lr, momentum, decay, nesterov)

def _build_model_residual_block(x, i, s):
    original_x = x

    tanh_out = CausalDilatedConv1D(settings.filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                   bias=settings.use_bias, name='dilated_conv_%d_tanh_s%d' % (2 ** i, s), activation='tanh',
                                   W_regularizer=l2(settings.res_l2))(x)
    sigm_out = CausalDilatedConv1D(settings.filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                   bias=settings.use_bias, name='dilated_conv_%d_sigm_s%d' % (2 ** i, s),
                                   activation='sigmoid', W_regularizer=l2(settings.res_l2))(x)
    x = layers.Multiply()([tanh_out, sigm_out])

    res_x = layers.Conv1D(settings.filters, 1, padding='same', use_bias=settings.use_bias, kernel_regularizer=l2(settings.res_l2))(x)
    skip_x = layers.Conv1D(settings.filters, 1, padding='same', use_bias=settings.use_bias, kernel_regularizer=l2(settings.res_l2))(x)
    res_x = layers.Add()([original_x, res_x])
    return res_x, skip_x


def _build_model():
    fragment_length = settings.frame_size
    input_shape = Input(shape=(fragment_length, settings.output_bins), name='input_part')

    out = input_shape
    skip_connections = []
    out = CausalDilatedConv1D(settings.filters, 2, atrous_rate=1, border_mode='valid', causal=True,
                              name='initial_causal_conv')(out)
    for s in range(settings.stacks):
        for i in range(0, settings.dilation_depth + 1):
            out, skip_out = _build_model_residual_block(out, i, s)
            skip_connections.append(skip_out)

    out = layers.Add()(skip_connections)
    out = layers.Activation('relu')(out)
    out = layers.Conv1D(settings.output_bins, 1, padding='same', kernel_regularizer=l2(settings.final_l2))(out)
    out = layers.Activation('relu')(out)
    out = layers.Conv1D(settings.output_bins, 1, padding='same')(out)

    out = layers.Activation('softmax', name="output_softmax")(out)
    model = Model(input_shape, out)
    # receptive_field, receptive_field_ms = _compute_receptive_field()
    return model


# def _compute_receptive_field():
#     return _compute_receptive_field2(settings.sample_rate, settings.dilation_depth, settings.stacks)
#
# def _compute_receptive_field2(sample_rate, dilation_depth, stacks):
#     receptive_field = stacks * (2 ** dilation_depth * 2) - (stacks - 1)
#     receptive_field_ms = (receptive_field * 1000) / sample_rate
#     return receptive_field, receptive_field_ms
