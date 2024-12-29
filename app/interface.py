"""
A Gradio-based interactive interface featuring three main tabs:
- Data Processing
- Model Training
- Inference

Provides visual controls and real-time feedback for users to perform
these operations through a clean, intuitive UI.
"""

import gradio as gr
from config.default import DEFAULT_CONFIG
from config.languages import LANG_JSON
from data.data import process_data
from trainer.trainer import train_model_generator, stop_training
from inference.inference import generate_text


def build_app_interface(selected_lang="en"):
    T = LANG_JSON[selected_lang]

    custom_css = """
    .gradio-container {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    .label-wrap .label-text {
        font-size: 14px !important;
        font-weight: 500 !important;
        display: block !important;
        margin-bottom: 8px !important;
    }
    .custom-log-container {
        border: 1px solid #ddd;
        border-radius: 4px;
        background: #ffffff;
        margin-top: 4px;
        padding: 8px;
    }
    #train-log-box {
        height: 150px !important;
        min-height: 150px !important;
        max-height: 150px !important;
        overflow-y: auto !important;
        font-family: monospace;
        padding: 8px;
        margin-bottom: 0 !important;
        background: white;
    }
    progress {
        width: 100%;
        height: 20px;
        margin: 4px 0;
    }
    .gap.compact {
        gap: 0.5rem !important;
    }
    """

    with gr.Blocks(title=T["app_title"], css=custom_css) as demo:
        # 语言选择下拉框
        lang_select = gr.Dropdown(
            choices=list(LANG_JSON.keys()),
            value=selected_lang,
            label=T["language_label"],
            interactive=True
        )

        with gr.Tabs() as main_tabs:
            # ------------------- data processing -------------------
            with gr.Tab(T["data_process_tab"]) as data_process_tab:
                with gr.Row():
                    input_text = gr.Textbox(label=T["dp_paste_text"], lines=19.5, placeholder="What's on your mind?")
                    with gr.Column():
                        txt_dir = gr.Textbox(label=T["dp_txt_dir"], value="")
                        with gr.Row():
                            raw_dir = gr.Textbox(label=T["dp_raw_dir"], value=DEFAULT_CONFIG["data_process"]["raw_data_dir"])
                            processed_dir = gr.Textbox(label=T["dp_processed_dir"], value=DEFAULT_CONFIG["data_process"]["processed_data_dir"])
                        with gr.Row():
                            no_val_set = gr.Checkbox(label=T["dp_no_val_set"], value=DEFAULT_CONFIG["data_process"]["no_validation"], interactive=True)
                            use_gpt2 = gr.Checkbox(label=T["dp_use_gpt2_tokenizer"], value=DEFAULT_CONFIG["data_process"]["use_gpt2_tokenizer"], interactive=True)
                        train_split = gr.Slider(label=T["dp_train_split"], minimum=0.1, maximum=0.99, step=0.01, value=DEFAULT_CONFIG["data_process"]["train_split_ratio"])
                        num_proc = gr.Number(label=T["dp_num_proc"], value=DEFAULT_CONFIG["data_process"]["num_proc"], precision=0, interactive=True)
                process_btn = gr.Button(T["dp_start_btn"])
                process_output = gr.Textbox(label=T["dp_result"], lines=5)

            # ------------------- training -------------------
            with gr.Tab(T["train_tab"]) as train_tab:
                train_params_title_md = gr.Markdown(f"### {T['train_params_title']}")

                with gr.Row():
                    data_dir_box = gr.Textbox(label=T["train_data_dir"], value=DEFAULT_CONFIG["training"]["data_dir"], interactive=True)
                    out_dir_box = gr.Textbox(label=T["train_out_dir"], value=DEFAULT_CONFIG["training"]["out_dir"], interactive=True)
                    backend_box = gr.Textbox(label=T["train_backend"], value=DEFAULT_CONFIG["training"]["backend"], interactive=True)
                    device_box = gr.Dropdown(
                        label=T["train_device"],
                        choices=["cpu", "cuda"],
                        value=DEFAULT_CONFIG["training"]["device"],
                        interactive=True
                    )
                    dtype_box = gr.Dropdown(
                        label=T["train_dtype"],
                        choices=["float16", "bfloat16", "float32"],
                        value=DEFAULT_CONFIG["training"]["dtype"],
                        interactive=True
                    )
                    compile_box = gr.Checkbox(label=T["train_compile_model"], value=DEFAULT_CONFIG["training"]["compile_model"], interactive=True)

                with gr.Row():
                    plot_interval_box = gr.Number(label=T["train_eval_interval"], value=DEFAULT_CONFIG["training"]["plot_interval"], interactive=True)
                    log_interval_box = gr.Number(label=T["train_log_interval"], value=DEFAULT_CONFIG["training"]["log_interval"], interactive=True)
                    num_eval_seeds_box = gr.Number(label=T["train_num_eval_seeds"], value=DEFAULT_CONFIG["training"]["num_eval_seeds"], interactive=True)
                    save_best_val_ckpt_box = gr.Checkbox(label=T["train_save_best_val_ckpt"], value=DEFAULT_CONFIG["training"]["save_best_val_checkpoint"], interactive=True)
                    init_from_box = gr.Dropdown(
                        label=T["train_init_from"],
                        choices=["scratch", "resume"],
                        value=DEFAULT_CONFIG["training"]["init_from"],
                        interactive=True
                    )
                    seed_box = gr.Number(label=T["train_seed"], value=DEFAULT_CONFIG["training"]["seed"], interactive=True)

                with gr.Row():
                    grad_acc_box = gr.Number(label=T["train_gas"], value=DEFAULT_CONFIG["training"]["gradient_accumulation_steps"], interactive=True)
                    batch_size_box = gr.Number(label=T["train_batch_size"], value=DEFAULT_CONFIG["training"]["batch_size"], interactive=True)
                    block_size_box = gr.Number(label=T["train_block_size"], value=DEFAULT_CONFIG["training"]["block_size"], interactive=True)
                    n_layer_box = gr.Number(label=T["train_n_layer"], value=DEFAULT_CONFIG["training"]["n_layer"], interactive=True)
                    n_head_box = gr.Number(label=T["train_n_head"], value=DEFAULT_CONFIG["training"]["n_head"], interactive=True)
                    n_embd_box = gr.Number(label=T["train_n_embd"], value=DEFAULT_CONFIG["training"]["n_embd"], interactive=True)

                with gr.Row():
                    dropout_box = gr.Number(label=T["train_dropout"], value=DEFAULT_CONFIG["training"]["dropout"], interactive=True)
                    bias_box = gr.Checkbox(label=T["train_bias"], value=DEFAULT_CONFIG["training"]["bias"], interactive=True)
                    lr_box = gr.Number(label=T["train_lr"], value=DEFAULT_CONFIG["training"]["learning_rate"], interactive=True)
                    max_iters_box = gr.Number(label=T["train_max_iters"], value=DEFAULT_CONFIG["training"]["max_iters"], interactive=True)
                    weight_decay_box = gr.Number(label=T["train_weight_decay"], value=DEFAULT_CONFIG["training"]["weight_decay"], interactive=True)

                with gr.Row():
                    beta1_box = gr.Number(label=T["train_beta1"], value=DEFAULT_CONFIG["training"]["beta1"], interactive=True)
                    beta2_box = gr.Number(label=T["train_beta2"], value=DEFAULT_CONFIG["training"]["beta2"], interactive=True)
                    lr_scheduler_box = gr.Dropdown(
                        label=T["train_lr_scheduler"],
                        choices=["none", "cosine", "constant_with_warmup", "linear", "step", "polynomial"],
                        value=DEFAULT_CONFIG["training"]["lr_scheduler_type"],
                        interactive=True
                    )
                    warmup_box = gr.Number(label=T["train_warmup_iters"], value=DEFAULT_CONFIG["training"]["warmup_iters"], interactive=True)
                    lr_decay_box = gr.Number(label=T["train_lr_decay_iters"], value=DEFAULT_CONFIG["training"]["lr_decay_iters"], interactive=True)
                    min_lr_box = gr.Number(label=T["train_min_lr"], value=DEFAULT_CONFIG["training"]["min_lr"], interactive=True)

                with gr.Row():
                    step_size_box = gr.Number(label="Step Size (for step decay)", value=DEFAULT_CONFIG["training"]["step_size"], interactive=True)
                    step_gamma_box = gr.Number(label="Step Gamma (for step decay)", value=DEFAULT_CONFIG["training"]["step_gamma"], interactive=True)
                    polynomial_power_box = gr.Number(label="Polynomial Power (for polynomial decay)", value=DEFAULT_CONFIG["training"]["polynomial_power"], interactive=True)
                    save_interval_box = gr.Number(label=T["train_save_interval"], value=DEFAULT_CONFIG["training"]["save_interval"], interactive=True)

                train_btn = gr.Button(T["train_start_btn"])
                stop_btn = gr.Button(T["stop_btn"])

                with gr.Row():
                    with gr.Column(scale=1):
                        train_progress = gr.HTML(label="Training Progress")
                        train_log = gr.Textbox(label=T["train_log"], elem_id="train-log-box", elem_classes="custom-log-container")
                    with gr.Column(scale=2):
                        train_plot = gr.Image(label=T["train_plot"], type="pil")

            # ------------------- inference -------------------
            with gr.Tab(T["infer_tab"]) as inf_tab:
                with gr.Row():
                    data_dir_inf = gr.Textbox(label=T["dp_processed_dir"], value=DEFAULT_CONFIG["inference"]["data_dir"])
                    out_dir_inf = gr.Textbox(label=T["inf_out_dir"], value="out/ckpt.pt") 
                prompt_box = gr.Textbox(label=T["inf_prompt"], value=DEFAULT_CONFIG["inference"]["prompt"], lines=5, placeholder="Just write something here!")
                with gr.Row():
                    num_samples_box = gr.Number(label=T["inf_num_samples"], value=DEFAULT_CONFIG["inference"]["num_samples"])
                    max_new_tokens_box = gr.Number(label=T["inf_max_new_tokens"], value=DEFAULT_CONFIG["inference"]["max_new_tokens"])
                    temperature_box = gr.Number(label=T["inf_temperature"], value=DEFAULT_CONFIG["inference"]["temperature"])
                    top_k_box = gr.Number(label=T["inf_top_k"], value=DEFAULT_CONFIG["inference"]["top_k"])
                    seed_box_inf = gr.Number(label=T["inf_seed"], value=DEFAULT_CONFIG["inference"]["seed"], precision=0, interactive=True)
                inf_btn = gr.Button(T["inf_start_btn"])
                inf_output = gr.Textbox(label=T["inf_result"], lines=10)

        # ------------------- data processing button callback -------------------
        def data_processing_cb(txt, ddir, rdir, pdir, sp, no_val, use_gpt2_tokenizer, num_proc_):
            try:
                info = process_data(
                    input_text=txt,
                    input_dir=ddir,
                    raw_data_dir=rdir,
                    processed_data_dir=pdir,
                    train_split_ratio=sp,
                    no_validation=no_val,
                    use_gpt2_tokenizer=use_gpt2_tokenizer,
                    num_proc=int(num_proc_)
                )
                msg = (
                    f"Processing complete! Data saved to {info['processed_data_dir']}.\n"
                    f"Vocabulary size: {info['vocab_size']}.\n"
                    f"Training set size: {info['train_size']}."
                )
                if 'val_size' in info and info['val_size'] is not None:
                    msg += f"\nVal size: {info['val_size']}."
                else:
                    msg += "\nNo validation set created."
                return msg
            except Exception as e:
                return f"Error: {str(e)}"

        process_btn.click(
            fn=data_processing_cb,
            inputs=[input_text, txt_dir, raw_dir, processed_dir, train_split, no_val_set, use_gpt2, num_proc],
            outputs=process_output
        )

        # ------------------- stop training button callback -------------------
        stop_btn.click(fn=stop_training, inputs=[], outputs=[])

        # ------------------- training button callback -------------------
        def training_cb(
            data_dir_, out_dir_, plot_interval_, log_interval_, num_eval_seeds_,
            save_best_val_ckpt_, init_from_, grad_acc_, batch_size_, block_size_,
            n_layer_, n_head_, n_embd_, dropout_, bias_,
            lr_, max_iters_, weight_decay_, beta1_, beta2_,
            lr_scheduler_type_, warmup_, lr_decay_, min_lr_,
            step_size_, step_gamma_, polynomial_power_,
            backend_, device_, dtype_, compile_,
            seed_, save_interval_
        ):
            img_pil = None
            try:
                num_eval_seeds_int = int(num_eval_seeds_)
                if num_eval_seeds_int < 0 or num_eval_seeds_int > 2**32 - 1:
                    raise ValueError("Seed out of range.")
            except ValueError as e:
                yield (f"<div style='color:red;'>{str(e)}</div>", str(e), img_pil)
                return

            try:
                seed_int = int(seed_)
                if not (0 <= seed_int <= 2**32 - 1):
                    raise ValueError("Seed out of range.")
            except ValueError as e:
                if num_eval_seeds_int == 0:
                    yield (f"<div style='color:red;'>{str(e)}</div>", str(e), img_pil)
                seed_int = 0

            try:
                save_interval_int = int(save_interval_)
                if save_interval_int < 0:
                    raise ValueError("Save interval must be a non-negative integer.")
            except ValueError as e:
                if num_eval_seeds_int == 0:
                    yield (f"<div style='color:red;'>{str(e)}</div>", str(e), img_pil)
                save_interval_int = DEFAULT_CONFIG["training"]["save_interval"]

            try:
                gen = train_model_generator(
                    data_dir=data_dir_,
                    out_dir=out_dir_,
                    plot_interval=int(plot_interval_),
                    log_interval=int(log_interval_),
                    num_eval_seeds=num_eval_seeds_int,
                    save_best_val_checkpoint=bool(save_best_val_ckpt_),
                    init_from=init_from_,
                    gradient_accumulation_steps=int(grad_acc_),
                    batch_size=int(batch_size_),
                    block_size=int(block_size_),
                    n_layer=int(n_layer_),
                    n_head=int(n_head_),
                    n_embd=int(n_embd_),
                    dropout=float(dropout_),
                    bias=bool(bias_),
                    learning_rate=float(lr_),
                    max_iters=int(max_iters_),
                    weight_decay=float(weight_decay_),
                    beta1=float(beta1_),
                    beta2=float(beta2_),
                    lr_scheduler_type=lr_scheduler_type_,
                    warmup_iters=int(warmup_),
                    lr_decay_iters=int(lr_decay_),
                    min_lr=float(min_lr_),
                    step_size=int(step_size_),
                    step_gamma=float(step_gamma_),
                    polynomial_power=float(polynomial_power_),
                    backend=backend_,
                    device=device_,
                    dtype=dtype_,
                    compile_model=bool(compile_),
                    seed=seed_int,
                    save_interval=save_interval_int
                )
                for (progress_html, log_html, img) in gen:
                    if isinstance(progress_html, str) and "Error" in progress_html:
                        error_html = f"<div style='color: red;'>{progress_html}</div>"
                        yield (error_html, log_html if log_html else "Error", img_pil)
                        return
                    yield (progress_html, log_html, img)
            except Exception as e:
                err_msg = f"An error occured: {str(e)}"
                err_html = f"<div style='color:red;'>{err_msg}</div>"
                yield (err_html, err_msg, img_pil)
                return

        train_btn.click(
            fn=training_cb,
            inputs=[
                data_dir_box, out_dir_box, plot_interval_box, log_interval_box, num_eval_seeds_box,
                save_best_val_ckpt_box, init_from_box, grad_acc_box, batch_size_box, block_size_box,
                n_layer_box, n_head_box, n_embd_box, dropout_box, bias_box,
                lr_box, max_iters_box, weight_decay_box, beta1_box, beta2_box,
                lr_scheduler_box, warmup_box, lr_decay_box, min_lr_box,
                step_size_box, step_gamma_box, polynomial_power_box,
                backend_box, device_box, dtype_box, compile_box,
                seed_box, save_interval_box
            ],
            outputs=[train_progress, train_log, train_plot]
        )

        # ------------------- inference button callback -------------------
        def inference_cb(
            data_dir_inf_, out_dir_inf_,
            prompt_, num_samples_, max_new_tokens_, temperature_, top_k_, seed_inf_
        ):
            try:
                num_samples_int = int(num_samples_)
                if num_samples_int <= 0 or num_samples_int > 1000:
                    yield "Error: Number of samples must be between 1 and 1000."
                    return
                accumulated_texts = [""] * num_samples_int

                gen = generate_text(
                    data_dir=data_dir_inf_,
                    out_dir=out_dir_inf_,
                    prompt=prompt_,
                    num_samples=num_samples_int,
                    max_new_tokens=int(max_new_tokens_),
                    temperature=float(temperature_),
                    top_k=int(top_k_) if top_k_ else None,
                    seed=int(seed_inf_),
                    device=DEFAULT_CONFIG["inference"]["device"],
                    dtype=DEFAULT_CONFIG["inference"]["dtype"],
                    compile_model=DEFAULT_CONFIG["inference"]["compile_model"]
                )

                current_sample_idx = 0
                for output in gen:
                    if output.startswith("Error"):
                        yield output
                        return
                    if output.startswith("Sample"):
                        parts = output.split(":\n", 1)
                        if len(parts) == 2:
                            sample_num = int(parts[0].split()[1]) - 1
                            text_content = parts[1]
                            if 0 <= sample_num < num_samples_int:
                                accumulated_texts[sample_num] = text_content
                    elif output.startswith("-" * 20):
                        current_sample_idx += 1

                    full_output = "\n\n".join([
                        f"Sample {i+1}:\n{text}"
                        for i, text in enumerate(accumulated_texts)
                        if text.strip()
                    ])
                    if full_output.strip():
                        yield full_output
                    else:
                        yield ""

                final_output = "\n\n".join([
                    f"Sample {i+1}:\n{text}"
                    for i, text in enumerate(accumulated_texts)
                    if text.strip()
                ])
                if final_output.strip():
                    yield final_output
                else:
                    yield "No text generated."
            except Exception as ex:
                yield f"An error occured: {str(ex)}"

        inf_btn.click(
            fn=inference_cb,
            inputs=[data_dir_inf, out_dir_inf, prompt_box, num_samples_box, max_new_tokens_box, temperature_box, top_k_box, seed_box_inf],
            outputs=inf_output
        )

        # ------------------- language switch callback -------------------
        def switch_language(lang_code):
            Tnew = LANG_JSON[lang_code]
            return [
                gr.update(value=lang_code, label=Tnew["language_label"]),
                gr.update(label=Tnew["data_process_tab"]),
                gr.update(label=Tnew["train_tab"]),
                gr.update(label=Tnew["infer_tab"]),
                gr.update(label=Tnew["dp_paste_text"]),
                gr.update(label=Tnew["dp_txt_dir"]),
                gr.update(label=Tnew["dp_raw_dir"], value=DEFAULT_CONFIG["data_process"]["raw_data_dir"]),
                gr.update(label=Tnew["dp_processed_dir"], value=DEFAULT_CONFIG["data_process"]["processed_data_dir"]),
                gr.update(label=Tnew["dp_train_split"], value=DEFAULT_CONFIG["data_process"]["train_split_ratio"]),
                gr.update(label=Tnew["dp_no_val_set"], value=DEFAULT_CONFIG["data_process"]["no_validation"]),
                gr.update(label=Tnew["dp_use_gpt2_tokenizer"], value=DEFAULT_CONFIG["data_process"]["use_gpt2_tokenizer"]),
                gr.update(label=Tnew["dp_num_proc"], value=DEFAULT_CONFIG["data_process"]["num_proc"]),
                gr.update(value=Tnew["dp_start_btn"]),
                gr.update(label=Tnew["dp_result"]),

                gr.update(value=f"### {Tnew['train_params_title']}", visible=True),
                gr.update(label=Tnew["train_data_dir"], value=DEFAULT_CONFIG["training"]["data_dir"]),
                gr.update(label=Tnew["train_out_dir"], value=DEFAULT_CONFIG["training"]["out_dir"]),
                gr.update(label=Tnew["train_eval_interval"], value=DEFAULT_CONFIG["training"]["plot_interval"]),
                gr.update(label=Tnew["train_log_interval"], value=DEFAULT_CONFIG["training"]["log_interval"]),
                gr.update(label=Tnew["train_num_eval_seeds"], value=DEFAULT_CONFIG["training"]["num_eval_seeds"]),
                gr.update(label=Tnew["train_save_best_val_ckpt"], value=DEFAULT_CONFIG["training"]["save_best_val_checkpoint"]),
                gr.update(label=Tnew["train_init_from"], value=DEFAULT_CONFIG["training"]["init_from"]),
                gr.update(label=Tnew["train_gas"], value=DEFAULT_CONFIG["training"]["gradient_accumulation_steps"]),
                gr.update(label=Tnew["train_batch_size"], value=DEFAULT_CONFIG["training"]["batch_size"]),
                gr.update(label=Tnew["train_block_size"], value=DEFAULT_CONFIG["training"]["block_size"]),
                gr.update(label=Tnew["train_n_layer"], value=DEFAULT_CONFIG["training"]["n_layer"]),
                gr.update(label=Tnew["train_n_head"], value=DEFAULT_CONFIG["training"]["n_head"]),
                gr.update(label=Tnew["train_n_embd"], value=DEFAULT_CONFIG["training"]["n_embd"]),
                gr.update(label=Tnew["train_dropout"], value=DEFAULT_CONFIG["training"]["dropout"]),
                gr.update(label=Tnew["train_bias"], value=DEFAULT_CONFIG["training"]["bias"]),
                gr.update(label=Tnew["train_lr"], value=DEFAULT_CONFIG["training"]["learning_rate"]),
                gr.update(label=Tnew["train_max_iters"], value=DEFAULT_CONFIG["training"]["max_iters"]),
                gr.update(label=Tnew["train_weight_decay"], value=DEFAULT_CONFIG["training"]["weight_decay"]),
                gr.update(label=Tnew["train_beta1"], value=DEFAULT_CONFIG["training"]["beta1"]),
                gr.update(label=Tnew["train_beta2"], value=DEFAULT_CONFIG["training"]["beta2"]),
                gr.update(label=Tnew["train_lr_scheduler"], value=DEFAULT_CONFIG["training"]["lr_scheduler_type"]),
                gr.update(label=Tnew["train_warmup_iters"], value=DEFAULT_CONFIG["training"]["warmup_iters"]),
                gr.update(label=Tnew["train_lr_decay_iters"], value=DEFAULT_CONFIG["training"]["lr_decay_iters"]),
                gr.update(label=Tnew["train_min_lr"], value=DEFAULT_CONFIG["training"]["min_lr"]),
                gr.update(label=Tnew["train_backend"], value=DEFAULT_CONFIG["training"]["backend"]),
                gr.update(label=Tnew["train_device"], value=DEFAULT_CONFIG["training"]["device"]),
                gr.update(label=Tnew["train_dtype"], value=DEFAULT_CONFIG["training"]["dtype"]),
                gr.update(label=Tnew["train_compile_model"], value=DEFAULT_CONFIG["training"]["compile_model"]),
                gr.update(value=Tnew["train_start_btn"]),
                gr.update(value=Tnew["stop_btn"]),
                gr.update(label=Tnew["train_log"]),
                gr.update(label=Tnew["train_plot"]),
                gr.update(label=Tnew["train_seed"], value=DEFAULT_CONFIG["training"]["seed"]),
                gr.update(label=Tnew["train_save_interval"], value=DEFAULT_CONFIG["training"]["save_interval"]),

                gr.update(label=Tnew["dp_processed_dir"], value=DEFAULT_CONFIG["inference"]["data_dir"]),
                gr.update(label=Tnew["inf_out_dir"], value=DEFAULT_CONFIG["inference"]["out_dir"]),
                gr.update(label=Tnew["inf_prompt"], value=DEFAULT_CONFIG["inference"]["prompt"]),
                gr.update(label=Tnew["inf_num_samples"], value=DEFAULT_CONFIG["inference"]["num_samples"]),
                gr.update(label=Tnew["inf_max_new_tokens"], value=DEFAULT_CONFIG["inference"]["max_new_tokens"]),
                gr.update(label=Tnew["inf_temperature"], value=DEFAULT_CONFIG["inference"]["temperature"]),
                gr.update(label=Tnew["inf_top_k"], value=DEFAULT_CONFIG["inference"]["top_k"]),
                gr.update(value=Tnew["inf_start_btn"]),
                gr.update(label=Tnew["inf_result"], value=""),
                gr.update(label=Tnew["inf_seed"], value=DEFAULT_CONFIG["inference"]["seed"])
            ]

        lang_select.change(
            fn=switch_language,
            inputs=[lang_select],
            outputs=[
                lang_select,
                data_process_tab, train_tab, inf_tab,
                input_text, txt_dir, raw_dir, processed_dir, train_split,
                no_val_set, use_gpt2, num_proc, process_btn, process_output,
                train_params_title_md,
                data_dir_box, out_dir_box, plot_interval_box,
                log_interval_box, num_eval_seeds_box,
                save_best_val_ckpt_box, init_from_box, grad_acc_box,
                batch_size_box, block_size_box, n_layer_box,
                n_head_box, n_embd_box, dropout_box, bias_box,
                lr_box, max_iters_box, weight_decay_box,
                beta1_box, beta2_box, lr_scheduler_box,
                warmup_box, lr_decay_box,
                min_lr_box, backend_box, device_box,
                dtype_box, compile_box, train_btn,
                stop_btn, train_log, train_plot,
                seed_box, save_interval_box,
                data_dir_inf, out_dir_inf, prompt_box,
                num_samples_box, max_new_tokens_box,
                temperature_box, top_k_box, inf_btn, inf_output,
                seed_box_inf
            ]
        )

    return demo

if __name__=="__main__":
    demo = build_app_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
