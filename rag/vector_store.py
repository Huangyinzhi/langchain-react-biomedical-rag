from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils.config_handler import faiss_conf
from model.factory import get_embed_model
embed_model = get_embed_model()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os
from dotenv import load_dotenv
load_dotenv()



class VectorStoreService:
    def __init__(self):
        # FAISS索引存储路径
        self.index_path = get_abs_path(faiss_conf["persist_directory"])
        # 判断索引文件夹是否存在
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            # 加载已有向量库
            self.vector_store = FAISS.load_local(
                folder_path=self.index_path,
                embeddings=embed_model,
                allow_dangerous_deserialization=True
            )
            logger.info("加载已存在FAISS向量索引")
        else:
            # 创建空向量库：先用一条空文档初始化FAISS
            self.vector_store = FAISS.from_documents(
                [Document(page_content="init")],
                embedding=embed_model
            )
            # 保存初始化索引到磁盘
            self.vector_store.save_local(self.index_path)
            logger.info("新建空白FAISS向量索引")

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=faiss_conf["chunk_size"],
            chunk_overlap=faiss_conf["chunk_overlap"],
            separators=faiss_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        # 检索接口完全兼容旧代码
        return self.vector_store.as_retriever(search_kwargs={"k": faiss_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入FAISS向量库
        计算文件的MD5做去重；分批提交Embedding，限制每批<=10条适配通义千问接口
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(faiss_conf["md5_hex_store"])):
                open(get_abs_path(faiss_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False
            with open(get_abs_path(faiss_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(faiss_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(faiss_conf["data_path"]),
            tuple(faiss_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue
            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue
                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # =========分批添加，每一批最大10条，解决DashScope接口限制========
                batch_size = 10
                for start_idx in range(0, len(split_document), batch_size):
                    batch = split_document[start_idx: start_idx + batch_size]
                    self.vector_store.add_documents(batch)

                # 持久化向量库到磁盘
                self.vector_store.save_local(self.index_path)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 内容加载成功")

            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
