import torch
import random
from torch.utils.data import Dataset

class CrossSubjectContrastiveTripletDataset(Dataset):
    def __init__(self, con_pairs, trip_pairs):
        """
        Contrastive + Triplet 학습이 가능한 Dataset

        Args:
            con_pairs (list of tuples): Contrastive Pair 데이터 리스트 [(x1, x2, label)]
            trip_pairs (list of tuples): Triplet Pair 데이터 리스트 [(anchor, positive, negative)]
        """
        self.con_pairs = con_pairs
        self.trip_pairs = trip_pairs

    def __len__(self):
        return max(len(self.con_pairs), len(self.trip_pairs))  # 최대 길이 기준으로 설정

    def __getitem__(self, idx):
        # Contrastive Pair 선택
        con_idx = idx % len(self.con_pairs)  # Index가 길이를 초과하지 않도록 보정
        x1, x2, contrastive_label = self.con_pairs[con_idx]

        # Triplet Pair 선택
        trip_idx = idx % len(self.trip_pairs)
        anchor, positive, negative = self.trip_pairs[trip_idx]
        y = anchor['volume'] # 성공한 사람의 경우의 volume을 예측하는 걱이 목표이기 때문
        
        ## SUBJNO, volume, result 제거하기 
        cols = [col for col in anchor.index if col not in ['SUBJNO', 'volume', 'result']] # ["AGE", "BMI", "AMH", "FSH"]
        return anchor[cols].to_numpy().astype('float32'), positive[cols].to_numpy().astype('float32'), negative[cols].to_numpy().astype('float32'), \
               x1[cols].to_numpy().astype('float32'), x2[cols].to_numpy().astype('float32'), contrastive_label, y.astype('float32')

def create_contrastive_pairs(data):
    pairs = []
    person_id = []
    for i in range(len(data)):
        x1 = data[i]
        label = data[i]['result'] # result, 임신 성공/실패 여부
        person_id.append(data[i]['SUBJNO']) # TRIPLET LOSS를 위햬
        
        # Positive Pair (같은 라벨)
        positive_idx = random.choice([j for j in range(len(data)) if data[j]['result'] == label and j != i])
        positive = data[positive_idx]
        pairs.append((x1, positive, 0))  # 같은 그룹이면 0 (가깝게)

        # Negative Pair (다른 라벨)
        negative_idx = random.choice([j for j in range(len(data)) if data[j]['result'] != label])
        negative = data[negative_idx]
        pairs.append((x1, negative, 1))  # 다른 그룹이면 1 (멀게)
    
    return pairs, person_id # pairs = (same, different label)


def create_triplet_data(data, person_ids):
    """
    Cross-subject Anchor 기반 Triplet Pair 생성

    Args:
        data (numpy array): 약물 용량을 포함한 입력 데이터
        labels (numpy array): 성공 (1) / 실패 (0) 레이블
        person_ids (list): 각 데이터 샘플이 어떤 사람(person)에 속하는지 표시하는 리스트

    Returns:
        triplets (list): (Anchor, Positive, Negative) 형태의 Triplet 데이터 리스트
    """

    triplets = []
    
    # 사람별 데이터 그룹화
    success_data = {pid: [] for pid in set(person_ids)}
    failure_data = {pid: [] for pid in set(person_ids)}

    for i, pid in enumerate(person_ids):
        if data[i]['result'] == 1:
            success_data[pid].append(data[i])
        else:
            failure_data[pid].append(data[i])

    # 성공한 사람들만 Triplet 생성 대상
    success_pids = [pid for pid in success_data if len(success_data[pid]) > 0]
    
    for anchor_pid in success_pids:
        for anchor_sample in success_data[anchor_pid]:  # 각 성공 데이터에 대해
        
            # Positive: 다른 사람의 성공 데이터에서 선택
            positive_pid = random.choice([pid for pid in success_pids if pid != anchor_pid])
            positive_sample = random.choice(success_data[positive_pid])
              
            # Negative: 같은 사람의 실패 데이터가 있으면 선택, 없으면 다른 사람의 실패 데이터 선택
            if len(failure_data[anchor_pid]) > 0:
                negative_sample = random.choice(failure_data[anchor_pid])
            else:
                negative_pid = random.choice([pid for pid in failure_data if len(failure_data[pid]) > 0])
                negative_sample = random.choice(failure_data[negative_pid])
            
            # Triplet 추가
            triplets.append((anchor_sample, positive_sample, negative_sample))

    return triplets


"""
Simple tabular dataset
X: row data
Y: class score
"""
class SimpTabDataset(Dataset):
    def __init__(self, data, selected_column, target):
        self.data = data 
        self.col = selected_column
        self.len = len(self.data)
        self.target = target
        self.sbj = 'SUBJNO'
        self.result = 'result' # 임신 성공, 실패
        
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        X = self.data[idx][self.col].to_numpy().astype('float32')  # 나이, BMI, AMH, FSH 
        y = self.data[idx][self.target].astype('int64') 

        return X, y, self.data[idx][self.sbj], self.data[idx][self.result]
