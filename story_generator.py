# 首先设置环境变量，确保在导入transformers之前生效
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"

# 然后导入其他模块
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# 设置模型存储目录
MODEL_DIR = "./local_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 加载中文预训练模型 - 全局加载，仅加载一次
try:
    # 尝试使用GPT2中文模型，添加国内镜像支持
    model_name = "uer/gpt2-chinese-cluecorpussmall"
    
    try:
        # 尝试从本地文件夹加载模型
        print(f"尝试从本地文件夹 {MODEL_DIR} 加载模型...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
        print("成功从本地文件夹加载模型")
    except Exception as local_e:
        print(f"从本地文件夹加载模型失败: {local_e}")
        print(f"尝试从国内镜像下载模型到 {MODEL_DIR}...")
        # 从镜像下载模型并保存到本地文件夹，显式指定镜像URL
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=MODEL_DIR,
            resume_download=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=MODEL_DIR,
            resume_download=True
        )
        
        # 保存模型到本地文件夹
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR)
        print(f"成功从国内镜像下载模型并保存到 {MODEL_DIR}")
    
    # 创建生成器
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    print("模型加载完成，生成器创建成功")
except Exception as e:
    print(f"模型加载过程中出现错误: {e}")
    # 降级使用更简单的模型
    generator = pipeline("text-generation", model="gpt2")
    print("成功加载备用模型")

# 辅助函数：去除生成文本中的编号列表
def remove_numbered_list(text):
    """去除文本中的编号列表，将编号转换为连续文本"""
    import re
    # 移除行首的数字编号（如 "1. "、"2. " 等）
    # 使用正则表达式匹配行首的数字+点+空格模式
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    # 移除重复的换行符，确保文本连续
    text = re.sub(r'\n+', '\n', text)
    return text

# 生成故事
def generate_story(keywords, genre, max_length=200, temperature=0.7):
    # 统一处理关键词分隔符，支持中文逗号和英文逗号
    keywords = keywords.replace('，', ',').strip()
    
    # 检测是否包含英文关键词
    if any(ord(c) < 128 and c.isalpha() for c in keywords):
        return "请使用中文关键词，生成英文故事暂不支持。"
    
    # 优化prompt，明确要求连续文本段落，避免编号列表
    prompt = f"请根据以下关键词生成一个{genre}风格的完整故事，要求以连续的文本段落形式呈现，不要使用数字编号列表，要有明确的开头、发展和结尾：{keywords}\n故事内容："
    
    try:
        result = generator(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            truncation=True,
            # 添加更多生成参数，减少编号生成
            pad_token_id=tokenizer.eos_token_id,  # 确保生成完整文本
            no_repeat_ngram_size=2,  # 避免重复
            num_return_sequences=1  # 只生成一个结果
        )
        
        story = result[0]["generated_text"].replace(prompt, "").strip()
        
        # 后处理：去除可能出现的编号列表
        story = remove_numbered_list(story)
        
        # 确保故事有完整结尾，避免截断
        if story and not any(story.endswith(punc) for punc in ['.', '。', '!', '！', '?', '？', '…', '…']):
            story += '。'
        return story
    except Exception as e:
        return f"生成故事时出错: {e}"

# 生成诗歌
def generate_poem(keywords, style="现代诗", max_length=100, temperature=0.8):
    # 统一处理关键词分隔符，支持中文逗号和英文逗号
    keywords = keywords.replace('，', ',').strip()
    
    try:
        # 根据诗歌风格设计不同的prompt模板
        if style == "现代诗":
            # 参考中国现代诗风格，要求意境优美，语言流畅
            prompt = f"请根据以下关键词创作一首优美的现代诗，要求以连续的分行形式呈现，不要使用任何数字编号，语言优美，意境深远，具有文学性：{keywords}\n诗歌内容："
        elif style == "古体诗":
            # 古体诗要求押韵，对仗工整
            prompt = f"请根据以下关键词创作一首古体诗，要求符合古诗格律，押韵工整，不要使用数字编号，语言典雅，意境优美：{keywords}\n诗歌内容："
        elif style == "宋词":
            # 宋词要求符合词牌格式，情感细腻
            prompt = f"请根据以下关键词创作一首宋词风格的作品，要求情感细腻，语言优美，不要使用数字编号，具有古典韵味：{keywords}\n诗歌内容："
        else: # 儿歌
            prompt = f"请根据以下关键词创作一首简单易懂的儿歌，要求语言明快，节奏流畅，不要使用数字编号，适合儿童传唱：{keywords}\n诗歌内容："
        
        result = generator(
            prompt,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.95,  # 增加多样性
            repetition_penalty=1.3,  # 减少重复
            do_sample=True,
            truncation=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # 避免重复短语
            num_return_sequences=1
        )
        
        poem = result[0]["generated_text"].replace(prompt, "").strip()
        
        # 增强后处理：去除编号列表
        poem = remove_numbered_list(poem)
        
        # 去除任何可能的数字编号（包括中文数字）
        import re
        poem = re.sub(r'^\s*[\d一二三四五六七八九十]+\s*[、.]\s*', '', poem, flags=re.MULTILINE)
        
        # 去除多余的换行符，确保诗歌分行合理
        poem = re.sub(r'\n+', '\n', poem)
        
        # 确保诗歌以换行符分隔，符合诗歌格式
        lines = poem.split('\n')
        # 过滤掉空行和只有空格的行
        lines = [line.strip() for line in lines if line.strip()]
        poem = '\n'.join(lines)
        
        # 为现代诗添加适当的分行
        if style == "现代诗" and len(lines) < 2:
            # 如果只有一行，尝试根据语义进行合理分行
            line = lines[0]
            # 按标点符号分行
            split_chars = ['，', '。', '！', '？', '；', '：']
            new_lines = []
            current_line = ''
            for char in line:
                current_line += char
                if char in split_chars:
                    new_lines.append(current_line.strip())
                    current_line = ''
            if current_line:
                new_lines.append(current_line.strip())
            if len(new_lines) > 1:
                poem = '\n'.join(new_lines)
        
        return poem
    except Exception as e:
        return f"生成诗歌时出错: {e}"

# 全局变量：保存历史记录和收藏内容
import json
import os
import time

# 历史记录和收藏文件路径
HISTORY_FILE = "generation_history.json"
FAVORITES_FILE = "favorites.json"

# 加载历史记录
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

# 保存历史记录
def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存历史记录失败: {e}")

# 加载收藏
def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

# 保存收藏
def save_favorites(favorites):
    try:
        with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
            json.dump(favorites, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存收藏失败: {e}")

# 初始化历史记录和收藏
history = load_history()
favorites = load_favorites()

# 创建Gradio界面
def create_interface():
    with gr.Blocks(
        title="AI故事/诗歌生成器",
        theme=gr.themes.Default(),  # 使用默认主题
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto;
        }
        .history-item {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .history-item:hover {
            background-color: #f5f5f5;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .favorite-btn {
            margin-top: 10px;
        }
        .export-btn {
            margin-top: 10px;
            margin-left: 10px;
        }
        .control-panel {
            background-color: #fafafa;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        .result-panel {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            min-height: 300px;
        }
        .tabs-container {
            margin-top: 20px;
        }
        .history-panel {
            max-height: 400px;
            overflow-y: auto;
            margin-top: 20px;
        }
        .keyword-buttons {
            margin-bottom: 20px;
        }
        .slider-label {
            margin-bottom: 5px;
            font-weight: 600;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }
        """
    ) as demo:
        # 页面标题和介绍
        gr.Markdown("# 🎨 AI故事/诗歌生成器")
        gr.Markdown("**智能创作，无限创意** - 输入关键词，生成属于你的精彩故事或优美诗歌")
        
        # 主内容区域
        with gr.Row():
            # 左侧控制面板
            with gr.Column(scale=1, min_width=400):
                # 功能选择标签页
                with gr.Tabs(elem_id="tabs-container") as tabs:
                    # 故事生成面板
                    with gr.TabItem("📖 故事生成", id="story-tab"):
                        gr.Markdown("## 故事生成", elem_classes="section-title")
                        
                        story_keywords = gr.Textbox(
                            label="🔑 关键词",
                            placeholder="输入关键词，用逗号分隔，如：公主,城堡,龙",
                            lines=2,
                            elem_classes="control-panel"
                        )
                        
                        # 故事关键词按钮组
                        gr.Markdown("### 常用关键词", elem_classes="slider-label")
                        with gr.Row(elem_classes="keyword-buttons"):
                            story_keyword_btns = [
                                gr.Button("公主", size="sm"),
                                gr.Button("城堡", size="sm"),
                                gr.Button("龙", size="sm"),
                                gr.Button("魔法", size="sm")
                            ]
                        with gr.Row(elem_classes="keyword-buttons"):
                            story_keyword_btns += [
                                gr.Button("冒险", size="sm"),
                                gr.Button("森林", size="sm"),
                                gr.Button("巫师", size="sm"),
                                gr.Button("宝藏", size="sm")
                            ]
                        
                        # 故事生成参数
                        with gr.Row():
                            with gr.Column():
                                story_theme = gr.Dropdown(
                                    choices=["奇幻", "科幻", "悬疑", "爱情", "冒险", "历史", "恐怖", "喜剧"],
                                    label="🎭 故事主题",
                                    value="奇幻",
                                    elem_classes="control-panel"
                                )
                                
                                story_style = gr.Dropdown(
                                    choices=["通俗", "文艺", "古典", "现代", "悬疑", "轻松"],
                                    label="✏️ 写作风格",
                                    value="通俗",
                                    elem_classes="control-panel"
                                )
                            
                            with gr.Column():
                                story_character = gr.Textbox(
                                    label="👤 主要角色",
                                    placeholder="如：勇敢的骑士、聪明的公主",
                                    elem_classes="control-panel"
                                )
                                
                                story_max_length = gr.Slider(
                                    minimum=100, 
                                    maximum=2000, 
                                    value=500, 
                                    label="📏 故事长度",
                                    step=50,
                                    elem_classes="control-panel"
                                )
                        
                        story_temperature = gr.Slider(
                            minimum=0.1, 
                            maximum=1.0, 
                            value=0.7, 
                            label="✨ 创意度",
                            step=0.1,
                            elem_classes="control-panel"
                        )
                        
                        generate_story_btn = gr.Button(
                            "🚀 生成故事",
                            variant="primary",
                            size="lg",
                            elem_classes="control-panel"
                        )
                    
                    # 诗歌生成面板
                    with gr.TabItem("📝 诗歌生成", id="poem-tab"):
                        gr.Markdown("## 诗歌生成", elem_classes="section-title")
                        
                        poem_keywords = gr.Textbox(
                            label="🔑 关键词",
                            placeholder="输入关键词，用逗号分隔，如：春天,花朵,希望",
                            lines=2,
                            elem_classes="control-panel"
                        )
                        
                        # 诗歌关键词按钮组
                        gr.Markdown("### 常用关键词", elem_classes="slider-label")
                        with gr.Row(elem_classes="keyword-buttons"):
                            poem_keyword_btns = [
                                gr.Button("春天", size="sm"),
                                gr.Button("花朵", size="sm"),
                                gr.Button("希望", size="sm"),
                                gr.Button("月光", size="sm")
                            ]
                        with gr.Row(elem_classes="keyword-buttons"):
                            poem_keyword_btns += [
                                gr.Button("梦想", size="sm"),
                                gr.Button("河流", size="sm"),
                                gr.Button("星辰", size="sm"),
                                gr.Button("思念", size="sm")
                            ]
                        
                        # 诗歌生成参数
                        with gr.Row():
                            with gr.Column():
                                poem_type = gr.Dropdown(
                                    choices=["现代诗", "古体诗", "宋词", "儿歌", "俳句", "自由诗"],
                                    label="📜 诗歌类型",
                                    value="现代诗",
                                    elem_classes="control-panel"
                                )
                                
                                poem_rhyme = gr.Dropdown(
                                    choices=["不要求", "押韵", "严格押韵", "偶句押韵"],
                                    label="🎵 押韵方式",
                                    value="不要求",
                                    elem_classes="control-panel"
                                )
                            
                            with gr.Column():
                                poem_lines = gr.Slider(
                                    minimum=4, 
                                    maximum=50, 
                                    value=12, 
                                    label="📏 行数控制",
                                    step=1,
                                    elem_classes="control-panel"
                                )
                                
                                poem_emotion = gr.Dropdown(
                                    choices=["喜悦", "忧伤", "思念", "励志", "平静", "激昂"],
                                    label="😊 情感基调",
                                    value="平静",
                                    elem_classes="control-panel"
                                )
                        
                        poem_temperature = gr.Slider(
                            minimum=0.1, 
                            maximum=1.0, 
                            value=0.8, 
                            label="✨ 创意度",
                            step=0.1,
                            elem_classes="control-panel"
                        )
                        
                        generate_poem_btn = gr.Button(
                            "🚀 生成诗歌",
                            variant="primary",
                            size="lg",
                            elem_classes="control-panel"
                        )
                    
                    # 历史记录面板
                    with gr.TabItem("📚 历史记录", id="history-tab"):
                        gr.Markdown("## 生成历史", elem_classes="section-title")
                        
                        history_list = gr.Dataset(
                            components=[gr.Textbox(label="标题"), gr.Textbox(label="内容"), gr.Textbox(label="类型")],
                            samples=history,
                            elem_id="history-panel"
                        )
                        
                        with gr.Row():
                            clear_history_btn = gr.Button("🗑️ 清空历史", variant="stop")
                            refresh_history_btn = gr.Button("🔄 刷新历史")
                    
                    # 收藏作品面板
                    with gr.TabItem("❤️ 我的收藏", id="favorites-tab"):
                        gr.Markdown("## 我的收藏", elem_classes="section-title")
                        
                        favorites_list = gr.Dataset(
                            components=[gr.Textbox(label="标题"), gr.Textbox(label="内容"), gr.Textbox(label="类型")],
                            samples=favorites,
                            elem_id="history-panel"
                        )
                        
                        with gr.Row():
                            remove_favorite_btn = gr.Button("🗑️ 移除收藏", variant="stop")
                            refresh_favorites_btn = gr.Button("🔄 刷新收藏")
                
            # 右侧结果展示区域
            with gr.Column(scale=2):
                gr.Markdown("## 🎯 生成结果", elem_classes="section-title")
                
                # 结果展示区域
                result_output = gr.Textbox(
                    label="",
                    lines=15,
                    interactive=False,
                    elem_classes="result-panel"
                )
                
                # 结果控制按钮
                with gr.Row():
                    favorite_btn = gr.Button("❤️ 收藏作品", variant="secondary")
                    export_btn = gr.Button("💾 导出文本", variant="secondary")
                    copy_btn = gr.Button("📋 复制内容", variant="secondary")
                    clear_result_btn = gr.Button("🗑️ 清空结果", variant="stop")
                
                # 导出文件组件
                export_file = gr.File(
                    label="下载文件",
                    visible=False
                )
        
        # 关键词按钮点击事件
        def add_keyword(textbox_value, keyword):
            if textbox_value.strip() == "":
                return keyword
            else:
                return f"{textbox_value.strip()},{keyword}"
        
        # 绑定故事关键词按钮
        for btn in story_keyword_btns:
            btn.click(
                fn=add_keyword,
                inputs=[story_keywords, gr.Textbox(value=btn.label, visible=False)],
                outputs=story_keywords
            )
        
        # 绑定诗歌关键词按钮
        for btn in poem_keyword_btns:
            btn.click(
                fn=add_keyword,
                inputs=[poem_keywords, gr.Textbox(value=btn.label, visible=False)],
                outputs=poem_keywords
            )
        
        # 故事生成函数包装器（带历史记录）
        def generate_story_with_history(keywords, genre, max_length, temperature):
            story = generate_story(keywords, genre, max_length, temperature)
            # 保存到历史记录
            global history
            history_item = {
                "title": f"故事_{time.strftime('%Y%m%d_%H%M%S')}",
                "content": story,
                "type": "故事",
                "timestamp": time.time(),
                "keywords": keywords,
                "genre": genre
            }
            history.append(history_item)
            # 只保留最近50条记录
            if len(history) > 50:
                history = history[-50:]
            save_history(history)
            return story
        
        # 诗歌生成函数包装器（带历史记录）
        def generate_poem_with_history(keywords, style, max_length, temperature):
            poem = generate_poem(keywords, style, max_length, temperature)
            # 保存到历史记录
            global history
            history_item = {
                "title": f"诗歌_{time.strftime('%Y%m%d_%H%M%S')}",
                "content": poem,
                "type": "诗歌",
                "timestamp": time.time(),
                "keywords": keywords,
                "style": style
            }
            history.append(history_item)
            # 只保留最近50条记录
            if len(history) > 50:
                history = history[-50:]
            save_history(history)
            return poem
        
        # 生成按钮事件
        generate_story_btn.click(
            fn=generate_story_with_history,
            inputs=[story_keywords, story_theme, story_max_length, story_temperature],
            outputs=result_output
        )
        
        generate_poem_btn.click(
            fn=generate_poem_with_history,
            inputs=[poem_keywords, poem_type, poem_lines, poem_temperature],  # 行数转换在函数内部处理
            outputs=result_output
        )
        
        # 收藏功能
        def add_to_favorites(content):
            if not content.strip():
                return "请先生成内容再收藏"
            global favorites
            favorite_item = {
                "title": f"收藏_{time.strftime('%Y%m%d_%H%M%S')}",
                "content": content,
                "type": "故事" if "故事" in content[:100] else "诗歌",
                "timestamp": time.time()
            }
            favorites.append(favorite_item)
            save_favorites(favorites)
            return "收藏成功！"
        
        favorite_btn.click(
            fn=add_to_favorites,
            inputs=[result_output],
            outputs=gr.Textbox(visible=False)
        )
        
        # 导出功能
        def export_content(content):
            if not content.strip():
                return None
            filename = f"ai_creation_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return filename
        
        export_btn.click(
            fn=export_content,
            inputs=[result_output],
            outputs=export_file
        )
        
        # 复制功能
        def copy_to_clipboard(content):
            import pyperclip
            pyperclip.copy(content)
            return "已复制到剪贴板！"
        
        copy_btn.click(
            fn=copy_to_clipboard,
            inputs=[result_output],
            outputs=gr.Textbox(visible=False)
        )
        
        # 清空结果
        clear_result_btn.click(
            fn=lambda: "",
            inputs=[],
            outputs=result_output
        )
        
        # 历史记录功能
        def refresh_history():
            global history
            history = load_history()
            return gr.Dataset.update(samples=history)
        
        refresh_history_btn.click(
            fn=refresh_history,
            inputs=[],
            outputs=history_list
        )
        
        def clear_history():
            global history
            history = []
            save_history(history)
            return gr.Dataset.update(samples=[])
        
        clear_history_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=history_list
        )
        
        # 收藏功能
        def refresh_favorites():
            global favorites
            favorites = load_favorites()
            return gr.Dataset.update(samples=favorites)
        
        refresh_favorites_btn.click(
            fn=refresh_favorites,
            inputs=[],
            outputs=favorites_list
        )
        
        # 移除收藏
        def remove_favorite(index):
            global favorites
            if 0 <= index < len(favorites):
                del favorites[index]
                save_favorites(favorites)
            return gr.Dataset.update(samples=favorites)
        
        remove_favorite_btn.click(
            fn=remove_favorite,
            inputs=[gr.Number(value=0, visible=False)],
            outputs=favorites_list
        )
        
        # 从历史记录加载内容
        def load_from_history(index):
            if 0 <= index < len(history):
                return history[index][1]  # 返回内容
            return ""
        
        history_list.click(
            fn=load_from_history,
            inputs=[history_list],
            outputs=result_output
        )
        
        # 从收藏加载内容
        def load_from_favorites(index):
            if 0 <= index < len(favorites):
                return favorites[index][1]  # 返回内容
            return ""
        
        favorites_list.click(
            fn=load_from_favorites,
            inputs=[favorites_list],
            outputs=result_output
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        theme=gr.themes.Default(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto;
        }
        .history-item {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .history-item:hover {
            background-color: #f5f5f5;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .favorite-btn {
            margin-top: 10px;
        }
        .export-btn {
            margin-top: 10px;
            margin-left: 10px;
        }
        .control-panel {
            background-color: #fafafa;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
        .result-panel {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            min-height: 300px;
        }
        .tabs-container {
            margin-top: 20px;
        }
        .history-panel {
            max-height: 400px;
            overflow-y: auto;
            margin-top: 20px;
        }
        .keyword-buttons {
            margin-bottom: 20px;
        }
        .slider-label {
            margin-bottom: 5px;
            font-weight: 600;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }
        """
    )
