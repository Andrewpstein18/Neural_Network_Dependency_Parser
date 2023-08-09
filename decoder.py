from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

import tensorflow as tf

from extract_training_data import FeatureExtractor, State

tf.compat.v1.disable_eager_execution()



class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
    
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)    

        while state.buffer:
            # extract an input vector for the current state
            features = np.array(self.extractor.get_input_representation(words, pos, state))
            features.reshape(1, 6)

            # get the result from the model about the softmax transitions
            softmax_result = self.model.predict(features)

            # sort in descending order the softmax results
            softmax_argsort = np.argsort(softmax_result)

            # before the loop, check if stack is empty
            if not state.stack:
                # shift
                state.shift()
                # continue to next iteration of the while loop
                continue


            # we know that stack is not empty
            # now check for legality
            for index in range(1, len(softmax_argsort[0])):

                # find the index of the greatest value
                maximum_transition_index = softmax_argsort[0, -1 * index]

                # find the associated action
                maximum_action_label = self.output_labels[maximum_transition_index]


                # Case 1: shift
                if maximum_action_label[0] == "shift":
                    # shifting the only word out of the buffer is illegal
                    if len(state.buffer) == 1:
                        continue
                    else:
                        state.shift()
                        # break out of for loop
                        break
                # Case 2: left_arc
                elif maximum_action_label[0] == "left_arc":
                    # left_arc towards the root is illegal
                    if state.stack[-1] == 0:
                        # check next highest value
                        continue
                    else:
                        state.left_arc(maximum_action_label[1])
                        # break out of for loop
                        break

                # Case 3: right_arc
                else:
                    state.right_arc(maximum_action_label[1])
                    # break out of for loop
                    break


        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()

        

