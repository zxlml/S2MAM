from model.models import main_cli, demo_r, demo_c
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # =============================================================================
    #    S2MAM 入口（可加参数，例如）：
    #    python main.py --task regression --dataset additive --engine implicit
    #    python main.py --task classification --dataset additive --engine implicit
    #    python main.py --task classification --dataset moon --engine reinforce
    # =============================================================================
    main_cli()
