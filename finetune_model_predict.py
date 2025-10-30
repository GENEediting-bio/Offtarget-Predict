# finetune_model_predict.py
"""
Nucleotide Transformer模型预测脚本 - 本地加载版本
完全避免网络连接，使用本地模型文件
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
import numpy as np
import json

# ---------------------
# 模型定义（与训练时相同）
# ---------------------
class NTClassificationModel(nn.Module):
    def __init__(self, backbone, num_labels=2, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        else:
            last_hidden = outputs[0]
            
        avg_pool = last_hidden.mean(dim=1)
        max_pool = last_hidden.max(dim=1)[0]
        pooled_output = avg_pool + max_pool
        
        logits = self.classifier(pooled_output)
        return logits

# ---------------------
# 本地模型加载函数
# ---------------------
def load_model_and_tokenizer_locally(model_path, device):
    """
    从本地目录加载模型和tokenizer，完全避免网络连接
    """
    print("🔧 正在从本地加载模型和tokenizer...")
    
    # 检查是否是HuggingFace模型目录
    if os.path.isdir(model_path):
        model_dir = model_path
    else:
        # 如果是检查点文件，尝试找到对应的模型目录
        checkpoint_dir = os.path.dirname(model_path)
        # 在检查点目录中查找模型文件
        possible_dirs = [
            os.path.join(checkpoint_dir, "model"),
            checkpoint_dir,
            os.path.dirname(checkpoint_dir)
        ]
        
        model_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(os.path.join(dir_path, "config.json")):
                model_dir = dir_path
                break
        
        if model_dir is None:
            print("❌ 未找到本地模型文件，请先下载模型到本地")
            print("请运行: python download_model.py 来下载模型")
            return None, None
    
    # 加载tokenizer
    try:
        # 首先尝试从本地加载
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        print("✅ Tokenizer本地加载成功")
    except Exception as e:
        print(f"❌ Tokenizer本地加载失败: {e}")
        print("请确保模型文件已完整下载到本地")
        return None, None
    
    # 加载模型配置
    try:
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        backbone = AutoModel.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            trust_remote_code=True
        )
        print("✅ 模型本地加载成功")
    except Exception as e:
        print(f"❌ 模型本地加载失败: {e}")
        return None, None
    
    return backbone, tokenizer

def download_model_locally(model_name="InstaDeepAI/nucleotide-transformer-500m-1000g", local_dir="./local_models"):
    """
    下载模型到本地目录的辅助函数
    """
    print(f"📥 正在下载模型 {model_name} 到本地目录 {local_dir}...")
    
    os.makedirs(local_dir, exist_ok=True)
    model_dir = os.path.join(local_dir, model_name.split('/')[-1])
    
    try:
        # 下载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=local_dir
        )
        tokenizer.save_pretrained(model_dir)
        print("✅ Tokenizer下载并保存成功")
        
        # 下载模型
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=local_dir
        )
        model.save_pretrained(model_dir)
        print("✅ 模型下载并保存成功")
        
        print(f"📁 模型已保存到: {model_dir}")
        return model_dir
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

# ---------------------
# 数据处理
# ---------------------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, tokenizer, max_length=512):
        self.seqs = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        
        enc = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item

def load_sequences_from_csv(csv_path):
    """从CSV文件加载序列"""
    df = pd.read_csv(csv_path)
    
    # 自动检测序列列
    sequence_columns = [col for col in df.columns if 'sequence' in col.lower() or 'seq' in col.lower() or 'dna' in col.lower()]
    if sequence_columns:
        seq_col = sequence_columns[0]
        print(f"检测到序列列: {seq_col}")
    else:
        # 如果没有找到序列相关的列名，使用第一列
        seq_col = df.columns[0]
        print(f"使用第一列作为序列: {seq_col}")
    
    sequences = df[seq_col].astype(str).tolist()
    return sequences, df

# ---------------------
# 预测函数
# ---------------------
def predict(model, dataloader, device):
    """进行预测"""
    model.eval()
    all_probs = []
    all_logits = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测中"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    
    return all_probs, all_logits, all_predictions

# ---------------------
# 模型加载
# ---------------------
def load_trained_model(checkpoint_path, local_model_dir, device):
    """加载训练好的模型（完全本地）"""
    print(f"加载模型检查点: {checkpoint_path}")
    
    # 加载backbone（本地）
    backbone, tokenizer = load_model_and_tokenizer_locally(local_model_dir, device)
    if backbone is None:
        return None, None
    
    # 创建模型架构
    model = NTClassificationModel(backbone, num_labels=2)
    
    # 加载训练好的权重
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✅ 模型权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return None, None
    
    model.to(device)
    model.eval()
    
    return model, tokenizer

# ---------------------
# 主函数
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="Nucleotide Transformer模型预测 - 本地版本")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="训练好的模型检查点路径")
    parser.add_argument("--input_csv", type=str, required=True,
                       help="输入数据CSV文件路径")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="预测结果输出CSV文件路径")
    parser.add_argument("--local_model_dir", type=str, 
                       default="./local_models/nucleotide-transformer-500m-1000g",
                       help="本地模型目录路径")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="预测批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    parser.add_argument("--device", type=str, default="cuda",
                       help="推理设备 (cuda/cpu)")
    parser.add_argument("--download_model", action="store_true",
                       help="如果本地没有模型，先下载模型")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"❌ 检查点文件不存在: {args.checkpoint}")
        return
    
    if not os.path.exists(args.input_csv):
        print(f"❌ 输入文件不存在: {args.input_csv}")
        return
    
    # 检查本地模型是否存在，如果不存在且指定了下载，则先下载
    if not os.path.exists(args.local_model_dir) and args.download_model:
        print("本地模型目录不存在，开始下载...")
        model_dir = download_model_locally(local_dir=os.path.dirname(args.local_model_dir))
        if model_dir is None:
            return
    elif not os.path.exists(args.local_model_dir):
        print(f"❌ 本地模型目录不存在: {args.local_model_dir}")
        print("请使用 --download_model 参数自动下载，或手动下载模型到该目录")
        return
    
    # 加载模型和tokenizer（完全本地）
    model, tokenizer = load_trained_model(args.checkpoint, args.local_model_dir, device)
    if model is None:
        print("❌ 模型加载失败")
        return
    
    # 加载数据
    print("正在加载数据...")
    sequences, original_df = load_sequences_from_csv(args.input_csv)
    print(f"加载了 {len(sequences)} 条序列")
    
    # 创建数据集和数据加载器
    dataset = SeqDataset(sequences, tokenizer, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 进行预测
    print("开始预测...")
    probabilities, logits, predictions = predict(model, dataloader, device)
    
    # 准备输出结果
    results_df = original_df.copy()
    
    # 添加预测结果
    results_df['prediction'] = predictions
    results_df['probability_class_0'] = [prob[0] for prob in probabilities]
    results_df['probability_class_1'] = [prob[1] for prob in probabilities]
    results_df['confidence'] = np.max(probabilities, axis=1)
    
    # 添加预测标签（可根据需要自定义）
    results_df['predicted_label'] = results_df['prediction'].map({0: 'negative', 1: 'positive'})
    
    # 保存结果
    results_df.to_csv(args.output_csv, index=False)
    print(f"✅ 预测完成！结果已保存到: {args.output_csv}")
    
    # 打印统计信息
    print("\n📊 预测统计:")
    print(f"   总样本数: {len(results_df)}")
    print(f"   预测为正类的样本数: {sum(predictions)}")
    print(f"   预测为负类的样本数: {len(predictions) - sum(predictions)}")
    print(f"   平均置信度: {np.mean(results_df['confidence']):.4f}")

# ---------------------
# 下载模型的独立脚本
# ---------------------
def download_model_script():
    """独立的模型下载脚本"""
    model_name = "InstaDeepAI/nucleotide-transformer-500m-1000g"
    local_dir = "./local_models"
    
    print("🚀 开始下载Nucleotide Transformer模型...")
    model_path = download_model_locally(model_name, local_dir)
    
    if model_path:
        print(f"\n🎉 模型下载完成！")
        print(f"📁 模型保存在: {model_path}")
        print(f"\n使用示例:")
        print(f"python finetune_model_predict.py --checkpoint your_checkpoint.pt --input_csv test.csv --output_csv predictions.csv --local_model_dir {model_path}")
    else:
        print("❌ 模型下载失败，请检查网络连接")

if __name__ == "__main__":
    # 如果没有参数，显示帮助信息
    import sys
    if len(sys.argv) == 1:
        print("Nucleotide Transformer本地预测脚本")
        print("\n使用方法:")
        print("1. 首先下载模型:")
        print("   python finetune_model_predict.py --download_model")
        print("\n2. 然后进行预测:")
        print("   python finetune_model_predict.py --checkpoint your_model.pt --input_csv test.csv --output_csv predictions.csv")
        print("\n3. 或者一步完成（自动下载模型）:")
        print("   python finetune_model_predict.py --checkpoint your_model.pt --input_csv test.csv --output_csv predictions.csv --download_model")
        sys.exit(1)
    
    main()