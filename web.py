import gradio as gr
import os
import cli

def process_file(file_obj):
    """
    接收文件路径作为输入，并返回视频文件的路径。
    """

    if file_obj is None:
        print("Error: 文件还在上传中")
        return "请等待文件上传完成"
    
    # 获取文件的临时路径
    file_path = file_obj.name
    # 确保路径正确地转换为字符串，尽管在大多数情况下这一步可能是多余的
    file_path_str = os.fspath(file_path)
    print(file_path_str)

    cfg_file = os.path.join(os.getcwd(), 'cli.ini')
    print('cfg_file:'+cfg_file)

    target_mp4 = cli.process(file_path_str, cfg_file)

    print(target_mp4)

    return target_mp4

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## 视频翻译工具")

        with gr.Row():
            file_input = gr.File(label="上传视频文件")  # 上传视频按钮

        with gr.Row():
            with gr.Column(scale=100):  # 左侧原视频
                original_video = gr.Video(label="原视频", elem_id="original-video")

            with gr.Column(scale=1, elem_id="button-column"):
                submit_button = gr.Button(value="一键翻译")  # 翻译按钮

            with gr.Column(scale=100):  # 右侧处理后的视频
                output_video = gr.Video(label="处理后的视频", elem_id="output-video")

        # 设置交互逻辑
        file_input.change(lambda file: file, inputs=file_input, outputs=original_video)
        submit_button.click(process_file, inputs=file_input, outputs=output_video)

    # 自定义CSS
    demo.css = """
        #original-video, #output-video {
            width: 100%;
            height: 400px;  # 设置视频组件的高度
        }
        #button-column {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    """

    # 启动界面
    demo.launch(debug=True, show_api=True)