def setup_args_Ours():
    train = argparse.ArgumentParser()
    train.add_argument("-model_type", "--model_type", type=str, default='Ours')
    train.add_argument("-exp_name", "--exp_name", type=str, default='modelv1')

    # about train setting
    train.add_argument("-batch_size", "--batch_size", type=int,
                       default=8)  # todo
    train.add_argument("-lr_bert", "--lr_bert", type=float, default=1e-5)
    train.add_argument("-lr_sasrec", "--lr_sasrec", type=float, default=1e-3)
    train.add_argument("-epoch", "--epoch", type=int, default=500)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-gpu", "--gpu", type=str, default='1')
    train.add_argument('--do_eval', action='store_true')

    # about model setting
    train.add_argument("-load_dict","--load_dict",type=str,\
        default="../../pretrain_model/wwm_ext/", help='load model path')
    train.add_argument("-init_add",
                       "--init_add",
                       action="store_true",
                       default=False)
    train.add_argument("-model_save_path",
                       "--model_save_path",
                       type=str,
                       default='saved_model/{}')  # todo
    train.add_argument("-model_load_path",
                       "--model_load_path",
                       type=str,
                       default='saved_model/{}')  # todo
    # about dataset and data setting
    train.add_argument("--raw", action="store_true", default=False)

    train.add_argument("-train_data_file","--train_data_file",type=str,\
        default="../../data/train_data.pkl", help='train data path')
    train.add_argument("-valid_data_file","--valid_data_file",type=str,\
        default="../../data/valid_data.pkl", help='valid data path')
    train.add_argument("-test_data_file","--test_data_file",type=str,\
        default="../../data/test_data.pkl", help='test data path')

    train.add_argument("-max_c_length",
                       "--max_c_length",
                       type=int,
                       default=256)  # pad_size
    train.add_argument("-use_size", "--use_size", type=int,
                       default=-1)  # pad_size
    train.add_argument("-vocab_path","--vocab_path",type=str,\
        default="../../pretrain_model/wwm_ext/vocab.txt", help='vocabulary to initial tokenizer')

    # other
    train.add_argument('--log_path',
                       default='log/{}.log',
                       type=str,
                       required=False,
                       help='log path')  #todo

    # SASRec
    train.add_argument("--hidden_size", type=int, default=50, \
        help="hidden size of transformer model")
    train.add_argument("--num_hidden_layers", type=int, default=2, \
        help="number of layers")
    train.add_argument('--num_attention_heads', default=1, type=int)
    train.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    train.add_argument("--attention_probs_dropout_prob", type=float, \
        default=0.2, help="attention dropout p")
    train.add_argument("--hidden_dropout_prob", type=float, default=0.2, \
        help="hidden dropout p")
    train.add_argument("--initializer_range", type=float, default=0.02)
    train.add_argument('--max_seq_length', default=100, type=int)
    train.add_argument('--item_size', default=33834, type=int)  #

    train.add_argument("-sasrec_load_path","--sasrec_load_path",type=str,\
        default="sasrec_{}.pth", help='sasrec load path')
    train.add_argument("-load_sasrec",
                       "--load_sasrec",
                       action="store_true",
                       default=False)
    train.add_argument("-sasrec_save_path","--sasrec_save_path",type=str, \
        default='sasrec_{}.pth') # todo

    train.add_argument("-load_fusion",
                       "--load_fusion",
                       action="store_true",
                       default=False)
    train.add_argument("-fusion_load_path","--fusion_load_path",type=str, \
        default='fusion_save_path_{}.pth') # todo
    train.add_argument("-fusion_save_path","--fusion_save_path",type=str, \
        default='fusion_save_path_{}.pth') # todo
    train.add_argument("-load_exp_name","--load_exp_name",type=str, \
        default='v1')


    train.add_argument("-seed", "--seed", type=int, default=42)  # todo
    train.add_argument("--weight_decay",
                       type=float,
                       default=0.0000,
                       help="weight_decay of adam")
    train.add_argument("--adam_beta1",
                       type=float,
                       default=0.9,
                       help="adam first beta value")
    train.add_argument("--adam_beta2",
                       type=float,
                       default=0.99,
                       help="adam second beta value")
    return train


def setup_args_BERT():
    train = argparse.ArgumentParser()
    train.add_argument("-exp_name", "--exp_name", type=str, default='modelv1')

    train.add_argument('--do_eval', action='store_true')
    train.add_argument("-batch_size", "--batch_size", type=int, default=8)
    train.add_argument("-lr_bert", "--lr_bert", type=float, default=1e-5)
    train.add_argument("-lr_sasrec", "--lr_sasrec", type=float, default=1e-3)
    train.add_argument("-epoch", "--epoch", type=int, default=500)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-gpu", "--gpu", type=str, default='2')
    train.add_argument("-max_c_length",
                       "--max_c_length",
                       type=int,
                       default=256)
    train.add_argument("-seed", "--seed", type=int, default=42)
    train.add_argument("--weight_decay",
                       type=float,
                       default=0.0000,
                       help="weight_decay of adam")
    train.add_argument("-use_size", "--use_size", type=int, default=-1)

    # about file path
    train.add_argument("-load_dict",
                       "--load_dict",
                       type=str,
                       default="../../pretrain_model/wwm_ext",
                       help='load model path')
    train.add_argument("-init_add",
                       "--init_add",
                       action="store_true",
                       default=False)
    train.add_argument("-model_save_path",
                       "--model_save_path",
                       type=str,
                       default='saved_model/bert_{}')

    # about dataset and data setting
    train.add_argument("-raw", "--raw", action="store_true", default=False)
    train.add_argument("-train_data_file",
                       "--train_data_file",
                       type=str,
                       default="../../data/train_data.pkl")
    train.add_argument("-valid_data_file",
                       "--valid_data_file",
                       type=str,
                       default="../../data/valid_data.pkl")
    train.add_argument("-test_data_file",
                       "--test_data_file",
                       type=str,
                       default="../../data/test_data.pkl")
    train.add_argument('--log_path',
                       default='log/{}.log',
                       type=str,
                       required=False,
                       help='log path')
    train.add_argument("-vocab_path",
                       "--vocab_path",
                       type=str,
                       default="../../pretrain_model/wwm_ext/vocab.txt",
                       help='vocabulary to initial tokenizer')

    return train


def setup_args_SASRec():
    train = argparse.ArgumentParser()
    train.add_argument("-exp_name", "--exp_name", type=str, default='modelv1')

    # about train setting
    train.add_argument("-batch_size", "--batch_size", type=int,
                       default=256)  # todo
    train.add_argument("-lr_bert", "--lr_bert", type=float, default=1e-5)
    train.add_argument("-lr_sasrec", "--lr_sasrec", type=float, default=1e-3)
    train.add_argument("-epoch", "--epoch", type=int, default=500)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-gpu", "--gpu", type=str, default='2')
    train.add_argument('--do_eval', action='store_true')
    train.add_argument("-seed", "--seed", type=int, default=42)  # todo

    # about model setting
    train.add_argument(
        "-load_dict",
        "--load_dict",
        type=str,
        default="../../pretrain_model/wwm_ext/",
    )
    train.add_argument("-init_add",
                       "--init_add",
                       action="store_true",
                       default=False)
    train.add_argument("-model_save_path",
                       "--model_save_path",
                       type=str,
                       default='saved_model/{}')  # todo

    # about dataset and data setting
    train.add_argument("-raw", "--raw", action="store_true", default=False)

    train.add_argument("-train_data_file",
                       "--train_data_file",
                       type=str,
                       default="../../data/train_data.pkl",
                       help='train data path')
    train.add_argument("-valid_data_file",
                       "--valid_data_file",
                       type=str,
                       default="../../data/valid_data.pkl",
                       help='valid data path')
    train.add_argument("-test_data_file",
                       "--test_data_file",
                       type=str,
                       default="../../data/test_data.pkl",
                       help='test data path')

    train.add_argument("-max_c_length",
                       "--max_c_length",
                       type=int,
                       default=256)  # pad_size
    train.add_argument("-use_size", "--use_size", type=int,
                       default=-1)  # pad_size
    train.add_argument("-vocab_path",
                       "--vocab_path",
                       type=str,
                       default="../../pretrain_model/wwm_ext/vocab.txt",
                       help='vocabulary to initial tokenizer')

    # other
    train.add_argument('--log_path',
                       default='log/{}.log',
                       type=str,
                       required=False,
                       help='log path')  #todo

    # SASRec
    train.add_argument("--hidden_size",
                       type=int,
                       default=50,
                       help="hidden size of transformer model")
    train.add_argument("--num_hidden_layers",
                       type=int,
                       default=2,
                       help="number of layers")
    train.add_argument('--num_attention_heads', default=1, type=int)
    train.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    train.add_argument("--attention_probs_dropout_prob",
                       type=float,
                       default=0.2,
                       help="attention dropout p")
    train.add_argument("--hidden_dropout_prob",
                       type=float,
                       default=0.2,
                       help="hidden dropout p")
    train.add_argument("--initializer_range", type=float, default=0.02)
    train.add_argument('--max_seq_length', default=60, type=int)
    train.add_argument('--item_size', default=33834, type=int)  #

    train.add_argument("-sasrec_load_path",
                       "--sasrec_load_path",
                       type=str,
                       default="0526.pth")
    train.add_argument("-load_sasrec",
                       "--load_sasrec",
                       action="store_true",
                       default=False)
    train.add_argument("-sasrec_save_path",
                       "--sasrec_save_path",
                       type=str,
                       default='sasrec_{}.pth')  # todo

    train.add_argument("-load_fusion",
                       "--load_fusion",
                       action="store_true",
                       default=False)
    # train.add_argument("-load_fusion_name","--load_fusion_name",type=str,default='') # todo
    train.add_argument("-fusion_load_path",
                       "--fusion_load_path",
                       type=str,
                       default='fusion_save_path_{}.pth')  # todo
    train.add_argument("-fusion_save_path",
                       "--fusion_save_path",
                       type=str,
                       default='fusion_save_path_{}.pth')  # todo

    # SASREC UNIQUE
    train.add_argument("--weight_decay",
                       type=float,
                       default=0.0000,
                       help="weight_decay of adam")
    train.add_argument("--adam_beta1",
                       type=float,
                       default=0.9,
                       help="adam first beta value")
    train.add_argument("--adam_beta2",
                       type=float,
                       default=0.99,
                       help="adam second beta value")

    return train
