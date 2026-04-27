from __future__ import annotations

import argparse

from compliant_mechanism_synthesis.training import (
    FlowCurriculumTrainingConfig,
    run_flow_training,
)
from compliant_mechanism_synthesis.utils import timestamped_run_dir


def _build_parser() -> argparse.ArgumentParser:
    defaults = FlowCurriculumTrainingConfig(dataset_path="")
    parser = argparse.ArgumentParser(
        prog="cms-train",
        description=(
            "Train the refiner with a unified rollout-based flow curriculum that transitions "
            "from local flow supervision toward physics-driven objectives."
        ),
    )
    parser.set_defaults(use_style_conditioning=defaults.use_style_conditioning)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--log-every-steps", type=int, default=defaults.log_every_steps)
    parser.add_argument(
        "--eval-every-steps", type=int, default=defaults.eval_every_steps
    )
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument(
        "--main-grad-clip-norm",
        type=float,
        default=defaults.main_grad_clip_norm,
    )
    parser.add_argument(
        "--physical-grad-clip-norm",
        type=float,
        default=defaults.physical_grad_clip_norm,
    )
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument(
        "--min-learning-rate", type=float, default=defaults.min_learning_rate
    )
    parser.add_argument("--eval-fraction", type=float, default=defaults.eval_fraction)
    parser.add_argument(
        "--num-integration-steps", type=int, default=defaults.num_integration_steps
    )
    parser.add_argument(
        "--flow-target-epsilon", type=float, default=defaults.flow_target_epsilon
    )
    parser.add_argument(
        "--max-position-step", type=float, default=defaults.max_position_step
    )
    parser.add_argument(
        "--max-adjacency-logit-step",
        type=float,
        default=defaults.max_adjacency_logit_step,
    )
    parser.add_argument(
        "--max-edge-radius-logit-step",
        type=float,
        default=defaults.max_edge_radius_logit_step,
    )
    parser.add_argument(
        "--style-token-dropout", type=float, default=defaults.style_token_dropout
    )
    parser.add_argument(
        "--no-style-conditioning",
        dest="use_style_conditioning",
        action="store_false",
        help="Disable oracle style conditioning during training.",
    )
    parser.add_argument(
        "--style-kl-loss-weight", type=float, default=defaults.style_kl_loss_weight
    )
    parser.add_argument(
        "--style-kl-anneal-steps", type=int, default=defaults.style_kl_anneal_steps
    )
    parser.add_argument(
        "--position-loss-weight", type=float, default=defaults.position_loss_weight
    )
    parser.add_argument(
        "--adjacency-loss-weight", type=float, default=defaults.adjacency_loss_weight
    )
    parser.add_argument(
        "--edge-radius-loss-weight",
        type=float,
        default=defaults.edge_radius_loss_weight,
    )
    parser.add_argument(
        "--position-huber-beta", type=float, default=defaults.position_huber_beta
    )
    parser.add_argument(
        "--adjacency-huber-beta", type=float, default=defaults.adjacency_huber_beta
    )
    parser.add_argument(
        "--edge-radius-huber-beta",
        type=float,
        default=defaults.edge_radius_huber_beta,
    )
    parser.add_argument(
        "--supervised-weight-start",
        type=float,
        default=defaults.supervised_weight_start,
    )
    parser.add_argument(
        "--supervised-weight-end", type=float, default=defaults.supervised_weight_end
    )
    parser.add_argument(
        "--supervised-transition-start-step",
        type=int,
        default=defaults.supervised_transition_start_step,
    )
    parser.add_argument(
        "--supervised-transition-end-step",
        type=int,
        default=defaults.supervised_transition_end_step,
    )
    parser.add_argument(
        "--physical-weight-start", type=float, default=defaults.physical_weight_start
    )
    parser.add_argument(
        "--physical-weight-end", type=float, default=defaults.physical_weight_end
    )
    parser.add_argument(
        "--physical-transition-start-step",
        type=int,
        default=defaults.physical_transition_start_step,
    )
    parser.add_argument(
        "--physical-transition-end-step",
        type=int,
        default=defaults.physical_transition_end_step,
    )
    parser.add_argument(
        "--stiffness-loss-weight", type=float, default=defaults.stiffness_loss_weight
    )
    parser.add_argument(
        "--stress-loss-weight", type=float, default=defaults.stress_loss_weight
    )
    parser.add_argument(
        "--material-loss-weight", type=float, default=defaults.material_loss_weight
    )
    parser.add_argument(
        "--short-beam-penalty-weight",
        type=float,
        default=defaults.short_beam_penalty_weight,
    )
    parser.add_argument(
        "--long-beam-penalty-weight",
        type=float,
        default=defaults.long_beam_penalty_weight,
    )
    parser.add_argument(
        "--thin-beam-penalty-weight",
        type=float,
        default=defaults.thin_beam_penalty_weight,
    )
    parser.add_argument(
        "--thick-beam-penalty-weight",
        type=float,
        default=defaults.thick_beam_penalty_weight,
    )
    parser.add_argument(
        "--free-node-spacing-penalty-weight",
        type=float,
        default=defaults.free_node_spacing_penalty_weight,
    )
    parser.add_argument(
        "--allowable-von-mises", type=float, default=defaults.allowable_von_mises
    )
    parser.add_argument(
        "--stress-activation-threshold",
        type=float,
        default=defaults.stress_activation_threshold,
    )
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument(
        "--simulation-position-grad-clip-norm",
        type=float,
        default=defaults.simulation_position_grad_clip_norm,
    )
    parser.add_argument(
        "--simulation-adjacency-grad-clip-norm",
        type=float,
        default=defaults.simulation_adjacency_grad_clip_norm,
    )
    parser.add_argument(
        "--simulation-edge-radius-grad-clip-norm",
        type=float,
        default=defaults.simulation_edge_radius_grad_clip_norm,
    )
    parser.add_argument(
        "--absolute-physical-loss-weight",
        type=float,
        default=defaults.absolute_physical_loss_weight,
    )
    parser.add_argument(
        "--relative-physical-loss-weight",
        type=float,
        default=defaults.relative_physical_loss_weight,
    )
    parser.add_argument("--init-checkpoint-path", default=defaults.init_checkpoint_path)
    parser.add_argument("--checkpoint-path", default=defaults.checkpoint_path)
    parser.add_argument("--logdir", default=defaults.logdir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--name", "-n", default="train")
    parser.add_argument(
        "--log-gradient-diagnostics",
        action="store_true",
        help="Log approximate supervised and physical gradient norms.",
    )
    return parser


def train_main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logdir_path = timestamped_run_dir(args.logdir, args.name)
    run_flow_training(
        FlowCurriculumTrainingConfig(
            dataset_path=args.dataset_path,
            device=args.device,
            batch_size=args.batch_size,
            log_every_steps=args.log_every_steps,
            eval_every_steps=args.eval_every_steps,
            max_grad_norm=args.max_grad_norm,
            main_grad_clip_norm=args.main_grad_clip_norm,
            physical_grad_clip_norm=args.physical_grad_clip_norm,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            min_learning_rate=args.min_learning_rate,
            eval_fraction=args.eval_fraction,
            init_checkpoint_path=args.init_checkpoint_path,
            checkpoint_path=args.checkpoint_path,
            logdir=str(logdir_path),
            seed=args.seed,
            num_integration_steps=args.num_integration_steps,
            flow_target_epsilon=args.flow_target_epsilon,
            max_position_step=args.max_position_step,
            max_adjacency_logit_step=args.max_adjacency_logit_step,
            max_edge_radius_logit_step=args.max_edge_radius_logit_step,
            use_style_conditioning=args.use_style_conditioning,
            style_token_dropout=args.style_token_dropout,
            style_kl_loss_weight=args.style_kl_loss_weight,
            style_kl_anneal_steps=args.style_kl_anneal_steps,
            position_loss_weight=args.position_loss_weight,
            adjacency_loss_weight=args.adjacency_loss_weight,
            edge_radius_loss_weight=args.edge_radius_loss_weight,
            position_huber_beta=args.position_huber_beta,
            adjacency_huber_beta=args.adjacency_huber_beta,
            edge_radius_huber_beta=args.edge_radius_huber_beta,
            supervised_weight_start=args.supervised_weight_start,
            supervised_weight_end=args.supervised_weight_end,
            supervised_transition_start_step=args.supervised_transition_start_step,
            supervised_transition_end_step=args.supervised_transition_end_step,
            physical_weight_start=args.physical_weight_start,
            physical_weight_end=args.physical_weight_end,
            physical_transition_start_step=args.physical_transition_start_step,
            physical_transition_end_step=args.physical_transition_end_step,
            stiffness_loss_weight=args.stiffness_loss_weight,
            stress_loss_weight=args.stress_loss_weight,
            material_loss_weight=args.material_loss_weight,
            short_beam_penalty_weight=args.short_beam_penalty_weight,
            long_beam_penalty_weight=args.long_beam_penalty_weight,
            thin_beam_penalty_weight=args.thin_beam_penalty_weight,
            thick_beam_penalty_weight=args.thick_beam_penalty_weight,
            free_node_spacing_penalty_weight=args.free_node_spacing_penalty_weight,
            allowable_von_mises=args.allowable_von_mises,
            stress_activation_threshold=args.stress_activation_threshold,
            weight_decay=args.weight_decay,
            simulation_position_grad_clip_norm=args.simulation_position_grad_clip_norm,
            simulation_adjacency_grad_clip_norm=args.simulation_adjacency_grad_clip_norm,
            simulation_edge_radius_grad_clip_norm=args.simulation_edge_radius_grad_clip_norm,
            absolute_physical_loss_weight=args.absolute_physical_loss_weight,
            relative_physical_loss_weight=args.relative_physical_loss_weight,
            log_gradient_diagnostics=args.log_gradient_diagnostics,
        )
    )
    print(f"logs={logdir_path}")
