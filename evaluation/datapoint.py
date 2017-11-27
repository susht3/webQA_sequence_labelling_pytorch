__all__ = ['EecommFeatures', 'Evidence', 'DataPoint']

class EecommFeatures(object):
    # field keys
    EECOMM_FEATURES = 'e-e.comm_features'
    OTHER_E_TYPE = 'other_evi_type'
    OTHER_E_KEY = 'other_evi_key'
    # optional
    OTHER_EVIDENCE_TOKENS = 'other_evi_tokens'

    @staticmethod
    def create(eecom_features, other_e_type, other_e_key,
            other_evi_tokens=None):
        '''
        @param eecom_features: a list of 0-1 labels
        @param other_e_type: the type of the other evidence used for computing
            the feature values, one of Evidence.POSITIVE,
            Evidence.HIT_ANS_NEGATIVE and Evidence.OTHER_NEGATIVE
        @param other_e_key: the key of the other evidence used for computing
            the feature values
        @param other_evi_tokens: optional, the tokens of the other evidence used
            for computing the feature values
        '''
        ret = {EecommFeatures.EECOMM_FEATURES:eecom_features,
               EecommFeatures.OTHER_E_TYPE:other_e_type,
               EecommFeatures.OTHER_E_KEY:other_e_key,}
        if other_evi_tokens is not None:
            ret[EecommFeatures.OTHER_EVIDENCE_TOKENS] = other_evi_tokens
        return ret


class Evidence(object):
    # field keys
    E_KEY = 'e_key'
    E_TOKENS = 'evidence_tokens'
    GOLDEN_LABELS = 'golden_labels'
    QECOMM_FEATURES = 'q-e.comm_features'
    GOLDEN_ANSWERS = 'golden_answers'
    TYPE = 'type' # seed
    SRC = 'source' # ANN - annotated, IR - retrieved
    EECOMM_FEATURES_LIST = 'eecom_features_list'

    # evidence types
    # the evidence is positive
    POSITIVE = 'positive'
    # the evidence is negative and contains the golden answer
    HIT_ANS_NEGATIVE = 'hit_answer_negative'
    # the evidence is negative and does not contain the golden answer
    OTHER_NEGATIVE = 'other_negative'

    # src
    ANNOTATED = 'ANN' # the evidence is annotated
    IR = 'IR' # the evidence is retrieved

    @staticmethod
    def create(e_key, e_tokens, golden_labels, qecomm_features, golden_answers,
               type_, src, eecomm_features_list=lambda: []):
        '''
        @param e_key: the key of this evidence
        @param e_tokens: evidence tokens, a list of strings
        @param golden_labels: BIO/BIO2 labels for each token of the evidence,
            a list of string
        @param qecomm_features: q-e.comm feature, a list of 0-1 values
        @param golden_answers: golden answers, a list of answers, each answer
            is a list of tokens
        @param type_: evidnece type, one of Evidence.POSITIVE,
            Evidence.HIT_ANS_NEGATIVE and Evidence.OTHER_NEGATIVE
        @param src: source of this evidence, one of Evidence.ANNOTATED or
            Evidence.IR
        @param eecomm_features_list: a list of EecommFeatures
        '''
        ret = {Evidence.E_KEY:e_key,
               Evidence.E_TOKENS:e_tokens,
               Evidence.QECOMM_FEATURES:qecomm_features,
               Evidence.GOLDEN_ANSWERS:golden_answers,
               Evidence.TYPE:type_,
               Evidence.SRC:src,
               Evidence.EECOMM_FEATURES_LIST:eecomm_features_list,}
        if golden_labels is not None:
            ret[Evidence.GOLDEN_LABELS] = golden_labels
        return ret


class DataPoint(object):
    # field keys
    Q_KEY = 'q_key'
    Q_TOKENS = 'question_tokens'
    EVIDENCES = 'evidences'

    @staticmethod
    def create(q_key, q_tokens, evidences):
        '''
        @param q_key: the key of the question
        @param q_tokens: question tokens, a list of strings
        @param evidences: a list of Evidence
        '''
        ret = {DataPoint.Q_KEY:q_key,
               DataPoint.Q_TOKENS:q_tokens,
               DataPoint.EVIDENCES:evidences,}
        return ret
