import tensorflow as tf
import tensorflow_text as tft
from transformers import TFBertModel, TFBertTokenizer

def BiLSTM(*args, **kwargs):
    return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(*args, **kwargs), merge_mode='sum')

class SweDeClinBert(tf.keras.Model):
    def __init__(self, num_classes:int, max_tokens:int=512, intermediate_size:int=768, dropout_rate:float=.2, activation=tf.nn.softmax, rnn:tf.keras.layers.Layer=tf.keras.layers.LSTM, **kwargs) -> None:
        super().__init__(**kwargs)

        # create tokenizer:
        #self.tokenizer = TFBertTokenizer.from_pretrained('./model/PyTorch/SweDeClin-BERT')
        self.tokenizer = TFBertTokenizer.from_pretrained('./model/TensorFlow/SweDeClin-BERT')
        self.tokenizer.padding    = "max_length"
        self.tokenizer.truncation = True
        self.tokenizer.max_length = max_tokens

        # create transformer:
        #self.transformer = TFBertModel.from_pretrained('./model/PyTorch/SweDeClin-BERT', from_pt=True)
        self.transformer = TFBertModel.from_pretrained('./model/TensorFlow/SweDeClin-BERT')

        # create output-layers:
        self.rnn     = rnn(intermediate_size, activation=tf.nn.tanh)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense   = tf.keras.layers.Dense(num_classes, activation=activation)

        # save sizes
        self.intermediate_size = intermediate_size
        self.output_size = num_classes


    def build(self, shape, optimized:bool=True):
        self.rnn.build((1 if optimized else shape[0], None if optimized else shape[1], 768))
        self.dropout.build((1 if optimized else shape[0], self.intermediate_size))
        self.dense.build((1 if optimized else shape[0], self.intermediate_size))

        self.built = True

    def get_config(self):
        config = super().get_config()

        config['num_classes'] = self.output_size
        config['max_tokens'] = self.tokenizer.max_length
        config['intermediate_size'] = self.intermediate_size
        config['dropout_rate'] = self.dropout.get_config()['rate']
        config['activation'] = self.dense.get_config()['activation']
        config['rnn'] = type(self.rnn)

        return config

    def compute_output_shape(self, input_shape):
        if not self.built: self.build(input_shape)
        return tf.TensorShape((input_shape[0] if len(input_shape) > 1 else 1, 1))

    def tokenize(self, inputs, return_token_type_ids:bool=True, return_attention_mask:bool=True) -> tf.Tensor:
        # Convert input to tensor if necessary:
        if isinstance(inputs, tf.RaggedTensor):
            inputs = inputs[:,::-1].to_tensor('')[:,::-1]

        elif not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)

        # Tokenize:
        return self.tokenizer(
            inputs,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask
        )

    def _call_bas(self, inputs, verbose) -> tf.Tensor:
        # Convert input to tensor if necessary:
        if isinstance(inputs, tf.RaggedTensor):
            inputs = inputs[:,::-1].to_tensor('')[:,::-1]

        elif not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)

        if verbose: print(inputs.shape)

        # flatten inputs:
        shape   = tf.shape(inputs)
        size    = tf.size(inputs)
        inputs  = tf.reshape(inputs, (size, 1))[:,0]

        if verbose: print(inputs.shape)

        # Tokenize if necessary:
        if inputs.dtype is tf.string:
            inputs = self.tokenizer(
                    inputs,
                    return_token_type_ids=True,
                    return_attention_mask=True
            )

        # propagate through transformer:
        outputs = self.transformer(inputs).pooler_output
        if verbose: print(outputs.shape)

        #reshape:
        shape = tf.tuple((shape[0], shape[1], tf.shape(outputs)[1]))
        outputs = tf.reshape(outputs, shape)
        if verbose: print(outputs.shape)

        # propagate through output layers:
        outputs = self.rnn(outputs)
        if verbose: print(outputs.shape)

        outputs = self.dropout(outputs)
        if verbose: print(outputs.shape)

        outputs = self.dense(outputs)
        if verbose: print(outputs.shape)

        return outputs

    def _call_opt(self, inputs, verbose) -> tf.Tensor:
        # Convert input to tensor if necessary:
        if isinstance(inputs, tf.RaggedTensor):
            inputs = inputs[:,::-1].to_tensor('')[:,::-1]

        elif not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs)

        if verbose: print(inputs.shape)

        # Tokenize if necessary:
        mask = (inputs != '')
        if inputs.dtype is tf.string:
            inputs = self.tokenizer(
                    tf.boolean_mask(inputs, mask),
                    return_token_type_ids=True,
                    return_attention_mask=True
            )

        # propagate through transformer:
        logits = self.transformer(inputs).pooler_output
        if verbose: print(logits.shape)

        # determine limits of transformer output:
        limits = tf.math.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        limits = tf.math.cumsum(limits)
        limits = tf.concat([tf.constant(0, shape=(1,), dtype=tf.int32), limits], axis=0)

        # propagate through output layers:
        def _propagate_sample(i, limits):
            sample = logits[tf.gather(limits, i):tf.gather(limits, i+1)]
            sample = tf.reshape(sample, tf.tuple([1, tf.shape(sample)[0], tf.shape(sample)[1]]))
            if verbose: print(sample.shape)

            sample = self.rnn(sample)
            if verbose: print(sample.shape)

            sample = self.dropout(sample)
            if verbose: print(sample.shape)

            sample = self.dense(sample)
            if verbose: print(sample.shape)

            return sample

        i = 0
        outputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)#, clear_after_read=False)
        outputs = tf.while_loop(
            cond = lambda i, _: tf.less(i+1, tf.size(limits)),
            body = lambda i, o: (i+1, o.write(i, _propagate_sample(i, limits)[0])),
            loop_vars = [i, outputs],
            parallel_iterations = 1,
            back_prop = True
        )[1]

        return outputs.stack()

    def call(self, inputs, optimized:bool=True, verbose:bool=False) -> tf.Tensor:
        if optimized: return self._call_opt(inputs, verbose)
        else:         return self._call_bas(inputs, verbose)
