import gradio as gr
from src.video_analysis import video_generator

iface = gr.Interface(
    fn=video_generator,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Image(label="Annotated Output"),
    live=True,
    title="Football Video Annotation",
    description="Upload an MP4 football video to see real-time inference and annotations without saving the result."
)

if __name__ == "__main__":
    iface.launch()