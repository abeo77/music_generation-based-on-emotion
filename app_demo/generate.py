
import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')
from dataloader import REMIFullSongTransformerDataset
from model.musemorphose import MuseMorphose

from utils import pickle_load, numpy_to_tensor, tensor_to_numpy
from remi2midi import remi2midi, remi2midi_NES

import torch
import yaml
import numpy as np
from scipy.stats import entropy

config_path = "config/default.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
vocab_path = config['data']['vocab_path']
data_split = 'pickles/test_pieces.pkl'

ckpt_path = "musemorphose_pretrained_weights.pt"
out_dir = "D:\out_path"
n_pieces = 1
n_samples_per_piece = 1

###########################################
# little helpers
###########################################
def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

def get_beat_idx(event):
  return int(event.split('_')[-1])

###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
  try:
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    assert np.count_nonzero(np.isnan(probs)) == 0
  except:
    print ('overflow detected, use 128-bit')
    logits = logits.astype(np.float128)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    probs = probs.astype(float)
  return probs

def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

########################################
# generation
########################################
def get_latent_embedding_fast(model, piece_data, use_sampling=False, sampling_var=0.):
  # reshape
  batch_inp = torch.from_numpy(piece_data['enc_input']).permute(1, 0).long().to(device)
  batch_padding_mask = torch.from_numpy(piece_data['enc_padding_mask']).bool().to(device)

  # get latent conditioning vectors
  with torch.no_grad():
    piece_latents = model.get_sampled_latent(
      batch_inp, padding_mask=batch_padding_mask, 
      use_sampling=use_sampling, sampling_var=sampling_var
    )

  return piece_latents

def generate_on_latent_ctrl_vanilla_truncate(
        model, latents, rfreq_cls, polyph_cls, event2idx, idx2event, 
        max_events=12800, primer=None,
        max_input_len=1280, truncate_len=512, 
        nucleus_p=0.9, temperature=1.2
      ):
  latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
  rfreq_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  polyph_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  print ('[info] rhythm cls: {} | polyph_cls: {}'.format(rfreq_cls, polyph_cls))

  if primer is None:
    generated = [event2idx['Bar_None']]
  else:
    generated = [event2idx[e] for e in primer]
    latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
    rfreq_placeholder[:len(generated), 0] = rfreq_cls[0]
    polyph_placeholder[:len(generated), 0] = polyph_cls[0]
    
  target_bars, generated_bars = latents.size(0), 0

  steps = 0
  time_st = time.time()
  cur_pos = 0
  failed_cnt = 0

  cur_input_len = len(generated)
  generated_final = deepcopy(generated)
  entropies = []

  while generated_bars < target_bars:
    if len(generated) == 1:
      dec_input = numpy_to_tensor([generated], device=device).long()
    else:
      dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

    latent_placeholder[len(generated)-1, 0, :] = latents[ generated_bars ]
    rfreq_placeholder[len(generated)-1, 0] = rfreq_cls[ generated_bars ]
    polyph_placeholder[len(generated)-1, 0] = polyph_cls[ generated_bars ]

    dec_seg_emb = latent_placeholder[:len(generated), :]
    dec_rfreq_cls = rfreq_placeholder[:len(generated), :]
    dec_polyph_cls = polyph_placeholder[:len(generated), :]

    # sampling
    with torch.no_grad():
      logits = model.generate(dec_input, dec_seg_emb, dec_rfreq_cls, dec_polyph_cls)
    logits = tensor_to_numpy(logits[0])
    probs = temperatured_softmax(logits, temperature)
    word = nucleus(probs, nucleus_p)
    word_event = idx2event[word]

    if 'Beat' in word_event:
      event_pos = get_beat_idx(word_event)
      if not event_pos >= cur_pos:
        failed_cnt += 1
        print ('[info] position not increasing, failed cnt:', failed_cnt)
        if failed_cnt >= 128:
          print ('[FATAL] model stuck, exiting ...')
          return generated
        continue
      else:
        cur_pos = event_pos
        failed_cnt = 0

    if 'Bar' in word_event:
      generated_bars += 1
      cur_pos = 0
      print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated_final)))
    if word_event == 'PAD_None':
      continue

    if len(generated) > max_events or (word_event == 'EOS_None' and generated_bars == target_bars - 1):
      generated_bars += 1
      generated.append(event2idx['Bar_None'])
      print ('[info] gotten eos')
      break

    generated.append(word)
    generated_final.append(word)
    entropies.append(entropy(probs))

    cur_input_len += 1
    steps += 1

    assert cur_input_len == len(generated)
    if cur_input_len == max_input_len:
      generated = generated[-truncate_len:]
      latent_placeholder[:len(generated)-1, 0, :] = latent_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0, :]
      rfreq_placeholder[:len(generated)-1, 0] = rfreq_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
      polyph_placeholder[:len(generated)-1, 0] = polyph_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]

      print ('[info] reset context length: cur_len: {}, accumulated_len: {}, truncate_range: {} ~ {}'.format(
        cur_input_len, len(generated_final), cur_input_len-truncate_len, cur_input_len-1
      ))
      cur_input_len = len(generated)

  assert generated_bars == target_bars
  print ('-- generated events:', len(generated_final))
  print ('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
  return generated_final[:-1], time.time() - time_st, np.array(entropies)


########################################
# change attribute classes
########################################
def random_shift_attr_cls(n_samples, upper=4, lower=-3):
  return np.random.randint(lower, upper, (n_samples,))


import torch
import numpy as np
from copy import deepcopy


def generate_custom_music(model, dset, p_data, device, n_samples_per_piece=1, max_input_len=1280, temperature=1.2,
                          nucleus_p=0.9, polyphony_range=(-3, 4), rhythmic_freq_range=(-3, 4)):
  # Lấy latent embedding nhanh cho bài nhạc
  p_latents = get_latent_embedding_fast(
    model, p_data,
    use_sampling=True,  # Sử dụng sampling cho việc sinh nhạc
    sampling_var=0.1  # Biến sampling, có thể điều chỉnh thêm
  )

  # Hàm tạo sự thay đổi ngẫu nhiên cho polyphony và rhythmic frequency
  def random_shift_attr_cls(n_samples, lower=-3, upper=4):
    return np.random.randint(lower, upper, (n_samples,))

  # Tạo sự thay đổi ngẫu nhiên cho polyphony và rhythmic frequency
  p_cls_diff = random_shift_attr_cls(n_samples_per_piece, *polyphony_range)
  r_cls_diff = random_shift_attr_cls(n_samples_per_piece, *rhythmic_freq_range)
  piece_entropies = []
  generated_songs = []

  # Chỉ tạo 1 mẫu nhạc cho mỗi bài
  for samp in range(n_samples_per_piece):
    # Tạo lớp polyphony và rhythmic frequency sau khi thay đổi
    p_polyph_cls = torch.from_numpy(p_data['polyph_cls_bar'] + p_cls_diff[samp]).clamp(0, 7).long()
    p_rfreq_cls = torch.from_numpy(p_data['rhymfreq_cls_bar'] + r_cls_diff[samp]).clamp(0, 7).long()

    # Chuẩn bị tên tệp để lưu kết quả (một tệp duy nhất)
    out_file = f"generated_piece_sample_{samp + 1}_poly_{p_cls_diff[samp]}_rhym_{r_cls_diff[samp]}"
    print(f"[info] Writing to: {out_file}")

    # Gọi hàm sinh nhạc
    song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
      model, p_latents, p_rfreq_cls, p_polyph_cls, dset.event2idx, dset.idx2event,
      max_input_len=max_input_len,
      nucleus_p=nucleus_p,
      temperature=temperature
    )

    # Thêm nhạc đã sinh vào danh sách kết quả
    generated_songs.append(song)
    piece_entropies.append(entropies.mean())

    # Chuyển đổi sự kiện sang định dạng nhạc và lưu thành tệp MIDI
    song = word2event(song, dset.idx2event)

    # Điều chỉnh enforce_tempo_val (nếu có) trong remi2midi
    enforce_tempo_val = None  # Bạn có thể tùy chỉnh giá trị tempo ở đây nếu cần
    remi2midi(song, "generated_music_output.mid", enforce_tempo=False, enforce_tempo_val=enforce_tempo_val)
    remi2midi_NES(song, 'generated_music_NES.mid', enforce_tempo=False, enforce_tempo_val=enforce_tempo_val)
    # Lưu siêu dữ liệu về nhạc đã sinh
    # np.save(out_file + '-POLYCLS.npy', tensor_to_numpy(p_polyph_cls))
    # np.save(out_file + '-RHYMCLS.npy', tensor_to_numpy(p_rfreq_cls))

    print(f"[info] Entropy: {entropies.mean():.4f} (+/- {entropies.std():.4f})")

  # Thống kê tổng hợp
  print(f"[time stats] Generation time: {np.mean(piece_entropies):.2f} secs (+/- {np.std(piece_entropies):.2f})")

  return generated_songs
import torch
import numpy as np
from remi2midi import remi2midi
from copy import deepcopy
dset = REMIFullSongTransformerDataset(
  data_dir, vocab_path,
  do_augment=False,
  model_enc_seqlen=config['data']['enc_seqlen'],
  model_dec_seqlen=config['generate']['dec_seqlen'],
  model_max_bars=config['generate']['max_bars'],
  pieces=pickle_load(data_split),
  pad_to_same=False
)
pieces = random.sample(range(len(dset)), n_pieces)
print ('[sampled pieces]', pieces)

mconf = config['model']
model = MuseMorphose(
  mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
  mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
  mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
  d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
  cond_mode=mconf['cond_mode']
).to(device)
model.eval()
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))