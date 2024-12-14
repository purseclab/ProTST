from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


if __name__ == "__main__":
    architecture_types = ['x86_32', 'x86_64', 'arm_32', 'arm_64', 'mips_32', 'mips_64', 'mipseb_32', 'mipseb_64']
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=special_tokens)
    tokenizer.train_from_iterator(architecture_types, trainer=trainer)
    tokenizer.save("workdir/1_prepare_pretrain_dataset/architecture_tokenizer.json")