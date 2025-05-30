import json
import requests
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def validate_documents(chunks):
    """Kiểm tra documents đã load thành công"""
    if not chunks:
        return False
    
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    return total_chars >= 1000

def validate_user_input(question):
    
    if not question or not question.strip():
        return False, "Vui lòng nhập câu hỏi"
    
    if len(question) > 500:
        return False, "Câu hỏi quá dài (tối đa 500 ký tự)"
    
    # Kiểm tra có chứa ký tự đặc biệt nguy hiểm
    dangerous_chars = ['<script>', '<?php', 'DROP TABLE']
    for char in dangerous_chars:
        if char.lower() in question.lower():
            return False, "Câu hỏi chứa nội dung không hợp lệ"
    
    return True, "Valid"

def validate_ollama_response(response_json):
    if not response_json:
        return False, "Response rỗng"
    
    if 'response' not in response_json:
        return False, "Thiếu field 'response'"
    
    content = response_json['response']
    
    if len(content) < 10:
        return False, "Response ngắn"
    
    return True, "Valid"

# Khởi động chatbot
try:
    prompt_config = json.load(open(r"C:\chatbot\prompt.json", 'r', encoding='utf-8'))
except FileNotFoundError:
    exit(1)
except json.JSONDecodeError as e:
    exit(1)

try:
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=100
    ).split_documents(
        UnstructuredFileLoader(r"C:\chatbot\data\tieuchuan2017.docx").load()
    )
    
    if not validate_documents(chunks):
        exit(1)
        
except Exception as e:
    exit(1)

# Tạo vector database
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
except Exception as e:
    exit(1)

def chat_with_ollama(question):
    try:
        # Tìm context liên quan
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        if not context.strip():
            return "Không tìm thấy thông tin liên quan trong tài liệu. Vui lòng thử câu hỏi khác."
        
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
        
        # Ollama API
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'qwen2:1.5b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.2, 
                    'top_p': 0.7,
                    'max_tokens': 1500
                }
            },
            timeout=40
        )
        
        if response.status_code != 200:
            return f"Lỗi Ollama (HTTP {response.status_code}): {response.text}"
        
        try:
            response_json = response.json()
        except json.JSONDecodeError:
            return "Lỗi: Ollama trả về response không phải JSON"
        
        is_valid, message = validate_ollama_response(response_json)
        if not is_valid:
            return f"Response không hợp lệ: {message}"
        
        result = response_json['response']
        
        if len(result) < 50:
            return f"Response ngắn bất thường: {result}"
        
        return result
        
    except requests.RequestException as e:
        return f"Lỗi kết nối Ollama: {e}"
    except Exception as e:
        return f"Lỗi hệ thống: {e}"

print(f"\nChatbot {prompt_config['context']['standard']['name']} sẵn sàng!")
print("Hỏi về: phân loại đèn, ghi nhãn, an toàn điện, phương pháp thử nghiệm, tiêu chuẩn")
print("Gõ 'exit' để thoát\n")

while True:
    question = input("Câu hỏi: ")
    
    if question.lower() in ['exit', 'quit', 'bye','thoát']:
        print("Tạm biệt!")
        break

    is_valid, validation_message = validate_user_input(question)
    if not is_valid:
        print(validation_message)
        continue

    answer = chat_with_ollama(question)
    print(f"\nChatbot: {answer}\n")
    print("-" * 80)