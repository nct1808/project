from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
from datetime import datetime
import uuid

# Import tất cả code từ file main.py của bạn
import requests
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)
CORS(app)

# ========== COPY TOÀN BỘ VALIDATION FUNCTIONS TỪ CODE CỦA BẠN ==========
def validate_prompt_config(config):
    """Kiểm tra cấu trúc prompt.json có đầy đủ không"""
    required_keys = [
        'system_prompt.role',
        'system_prompt.capabilities', 
        'response_format.structure',
        'context.standard.name',
        'instructions.context_usage'
    ]
    
    missing_keys = []
    
    for key in required_keys:
        keys = key.split('.')
        current = config
        
        try:
            for k in keys:
                current = current[k]
        except (KeyError, TypeError):
            missing_keys.append(key)
    
    if missing_keys:
        print(f" Lỗi cấu trúc prompt.json - thiếu: {missing_keys}")
        return False
    else:
        print(" prompt.json hợp lệ")
        return True

def validate_documents(chunks):
    """Kiểm tra documents đã load thành công"""
    if not chunks:
        print(" Không load được tài liệu!")
        return False
    
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    
    print(f" Load thành công {len(chunks)} chunks")
    print(f" Tổng cộng {total_chars:,} ký tự")
    
    if total_chars < 1000:
        print(" Cảnh báo: Tài liệu có vẻ quá ngắn")
    
    return True

def validate_user_input(question):
    """Kiểm tra input từ user"""
    if not question or not question.strip():
        return False, " Vui lòng nhập câu hỏi"
    
    if len(question) > 500:
        return False, " Câu hỏi quá dài (tối đa 500 ký tự)"
    
    # Kiểm tra có chứa ký tự đặc biệt nguy hiểm
    dangerous_chars = ['<script>', '<?php', 'DROP TABLE']
    for char in dangerous_chars:
        if char.lower() in question.lower():
            return False, " Câu hỏi chứa nội dung không hợp lệ"
    
    return True, "Valid"

def validate_ollama_response(response_json):
    """Kiểm tra response từ Ollama"""
    if not response_json:
        return False, "Response rỗng"
    
    if 'response' not in response_json:
        return False, "Thiếu field 'response'"
    
    content = response_json['response']
    
    if len(content) < 10:
        return False, "Response quá ngắn"
    
    # Kiểm tra có follow format không
    has_format_indicators = any(indicator in content for indicator in ['📋', '🔍', '⚠️', 'TCVN'])
    
    return True, f"Valid (có format: {has_format_indicators})"

# ========== GLOBAL VARIABLES ==========
prompt_config = None
retriever = None
chatbot_initialized = False
initialization_error = None

# Chat history management
HISTORY_FILE = "chat_history.json"
MAX_HISTORY_ITEMS = 1000

def load_chat_history():
    """Load chat history from file"""
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_chat_history(history):
    """Save chat history to file"""
    try:
        # Giới hạn số lượng lịch sử
        if len(history) > MAX_HISTORY_ITEMS:
            history = history[-MAX_HISTORY_ITEMS:]
        
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Lỗi lưu lịch sử: {e}")
        return False

def add_chat_to_history(question, answer, session_id=None):
    """Add new chat to history"""
    history = load_chat_history()
    
    chat_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id or "default",
        "question": question,
        "answer": answer,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S")
    }
    
    history.append(chat_entry)
    save_chat_history(history)
    return chat_entry

def get_chat_sessions():
    """Get all chat sessions grouped by date"""
    history = load_chat_history()
    sessions = {}
    
    for chat in reversed(history):  # Mới nhất trước
        date = chat.get('date', 'Unknown')
        if date not in sessions:
            sessions[date] = []
        sessions[date].append(chat)
    
    return sessions

def search_chat_history(query):
    """Search in chat history"""
    history = load_chat_history()
    results = []
    
    query_lower = query.lower()
    
    for chat in history:
        if (query_lower in chat.get('question', '').lower() or 
            query_lower in chat.get('answer', '').lower()):
            results.append(chat)
    
    return sorted(results, key=lambda x: x['timestamp'], reverse=True)

def delete_chat_history():
    """Clear all chat history"""
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return True
    except Exception as e:
        print(f"Lỗi xóa lịch sử: {e}")
        return False

def initialize_chatbot():
    """Khởi tạo chatbot - copy từ code gốc của bạn"""
    global prompt_config, retriever, chatbot_initialized, initialization_error
    
    try:
        print(" Đang khởi động chatbot...")
        
        # 1. Load và validate config
        print("Đang tải cấu hình...")
        try:
            prompt_config = json.load(open(r"C:\chatbot\prompt.json", 'r', encoding='utf-8'))
            
            if not validate_prompt_config(prompt_config):
                raise Exception("Lỗi cấu hình prompt.json")
                
        except FileNotFoundError:
            raise Exception("Không tìm thấy file prompt.json")
        except json.JSONDecodeError as e:
            raise Exception(f"Lỗi format JSON: {e}")

        # 2. Load và validate documents
        print(" Đang tải tài liệu...")
        try:
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1200, 
                chunk_overlap=100
            ).split_documents(
                UnstructuredFileLoader(r"C:\chatbot\data\tieuchuan2017.docx").load()
            )
            
            if not validate_documents(chunks):
                raise Exception("Lỗi tài liệu")
                
        except Exception as e:
            raise Exception(f"Lỗi load tài liệu: {e}")

        # 3. Tạo vector database
        print(" Đang tạo vector database...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={'k': 3})
            print(" Vector database sẵn sàng")
        except Exception as e:
            raise Exception(f"Lỗi tạo vector database: {e}")
        
        chatbot_initialized = True
        print(f"\n Chatbot {prompt_config['context']['standard']['name']} sẵn sàng!")
        
    except Exception as e:
        initialization_error = str(e)
        print(f" Lỗi khởi tạo: {e}")
        raise

def chat_with_ollama(question):
    """Copy nguyên xi function từ code của bạn"""
    try:
        # Tìm context liên quan
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Validate context tìm được
        if not context.strip():
            return " Không tìm thấy thông tin liên quan trong tài liệu. Vui lòng thử câu hỏi khác."
        
        # Tạo prompt theo format đã định
        prompt = f"""{prompt_config['system_prompt']['role']}

Khả năng:
{chr(10).join([f"- {c}" for c in prompt_config['system_prompt']['capabilities']])}

{prompt_config['instructions']['context_usage']}

Ngữ cảnh từ {prompt_config['context']['standard']['name']}:
{context}

Câu hỏi: {question}

Trả lời theo format:
{prompt_config['response_format']['structure']}

Trả lời:"""
        
        if len(prompt) > 8000: 
            print(f" Prompt dài ({len(prompt)} chars), có thể bị cắt")

        # Ollama API
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'qwen2:1.5b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.2, 
                    'top_p': 0.9,
                    'max_tokens': 1500
                }
            },
            timeout=30
        )
        
        # Validate HTTP response
        if response.status_code != 200:
            return f" Lỗi Ollama (HTTP {response.status_code}): {response.text}"
        
        # Validate JSON response
        try:
            response_json = response.json()
        except json.JSONDecodeError:
            return " Lỗi: Ollama trả về response không phải JSON"
        
        # Validate response content
        is_valid, message = validate_ollama_response(response_json)
        if not is_valid:
            return f" Response không hợp lệ: {message}"
        
        result = response_json['response']
        
        # Post-validation: Kiểm tra output quality
        if len(result) < 50:
            return f" Response ngắn bất thường: {result}"
        
        return result
        
    except requests.RequestException as e:
        return f" Lỗi kết nối Ollama: {e}"
    except Exception as e:
        return f" Lỗi hệ thống: {e}"

# ========== FLASK ROUTES ==========

@app.route('/')
def index():
    """Serve HTML interface"""
    # Đọc file app.html (bạn save file HTML ở cùng thư mục)
    try:
        with open('app.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return "File app.html không tìm thấy. Vui lòng đặt file app.html cùng thư mục với app_server.py"

@app.route('/api/status')
def get_status():
    """Kiểm tra status của chatbot"""
    return jsonify({
        'initialized': chatbot_initialized,
        'error': initialization_error
    })

@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Khởi tạo chatbot"""
    try:
        initialize_chatbot()
        return jsonify({
            'success': True,
            'message': 'Chatbot đã sẵn sàng!',
            'standard_name': prompt_config['context']['standard']['name'] if prompt_config else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint cho chat - sử dụng code gốc của bạn"""
    if not chatbot_initialized:
        return jsonify({
            'success': False,
            'error': 'Chatbot chưa được khởi tạo'
        }), 400
    
    data = request.json
    question = data.get('question', '')
    session_id = data.get('session_id', 'default')
    
    # Validate user input - dùng function của bạn
    is_valid, validation_message = validate_user_input(question)
    if not is_valid:
        return jsonify({
            'success': False,
            'error': validation_message
        })
    
    # Call chat function của bạn
    try:
        answer = chat_with_ollama(question)
        
        # Lưu vào lịch sử
        chat_entry = add_chat_to_history(question, answer, session_id)
        
        return jsonify({
            'success': True,
            'data': answer,
            'chat_id': chat_entry['id'],
            'timestamp': chat_entry['timestamp']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi xử lý: {str(e)}'
        })

@app.route('/api/history')
def get_history():
    """Lấy lịch sử chat"""
    try:
        sessions = get_chat_sessions()
        return jsonify({
            'success': True,
            'data': sessions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi lấy lịch sử: {str(e)}'
        })

@app.route('/api/history/search')
def search_history():
    """Tìm kiếm trong lịch sử"""
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Vui lòng nhập từ khóa tìm kiếm'
        })
    
    try:
        results = search_chat_history(query)
        return jsonify({
            'success': True,
            'data': results,
            'total': len(results)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi tìm kiếm: {str(e)}'
        })

@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    """Xóa toàn bộ lịch sử"""
    try:
        success = delete_chat_history()
        if success:
            return jsonify({
                'success': True,
                'message': 'Đã xóa toàn bộ lịch sử'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Không thể xóa lịch sử'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi xóa lịch sử: {str(e)}'
        })

@app.route('/api/history/stats')
def get_history_stats():
    """Thống kê lịch sử chat"""
    try:
        history = load_chat_history()
        
        stats = {
            'total_chats': len(history),
            'total_days': len(set(chat.get('date', '') for chat in history)),
            'latest_chat': history[-1]['timestamp'] if history else None,
            'first_chat': history[0]['timestamp'] if history else None
        }
        
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Lỗi thống kê: {str(e)}'
        })

@app.route('/api/validate', methods=['POST'])
def validate_api():
    """API để validate input trước khi gửi"""
    data = request.json
    question = data.get('question', '')
    
    is_valid, message = validate_user_input(question)
    
    return jsonify({
        'valid': is_valid,
        'message': message
    })

if __name__ == '__main__':
    print("🚀 Khởi động Flask server...")
    print("📝 Đang tự động khởi tạo chatbot...")
    
    # Auto-initialize khi start server
    try:
        initialize_chatbot()
        print("✅ Chatbot đã sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")
        print("⚠️  Server vẫn chạy nhưng chatbot chưa sẵn sàng")
    
    print("🌐 Mở browser và truy cập: http://localhost:5000")
    app.run(debug=True, port=5000)