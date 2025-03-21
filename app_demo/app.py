import os
from flask import Flask, render_template, request, send_file
import torch
import yaml
import numpy as np
import soundfile as sf
import generate # Import các mô-đun của bạn
import random
import pretty_midi
from remi2midi import remi2midi
import mido
from mido import MidiFile, MidiTrack, Message


def split_piano_to_p1_tr_reverse(input_midi, output_midi):
    midi = MidiFile(input_midi)
    new_midi = MidiFile()

    p1_track = MidiTrack()
    tr_track = MidiTrack()

    new_midi.tracks.append(p1_track)
    new_midi.tracks.append(tr_track)

    active_notes = []  # Lưu các nốt đang phát để phát hiện hợp âm

    for track in midi.tracks:
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes.append((msg.note, msg.time))

                    # Nếu có từ 2 nốt trở lên, tách thành P1 và TR
                    if len(active_notes) == 2:
                        lower_note = min(active_notes, key=lambda x: x[0])  # Nốt thấp hơn
                        higher_note = max(active_notes, key=lambda x: x[0])  # Nốt cao hơn

                        # Đưa nốt thấp vào Pulse 1 (P1)
                        p1_track.append(Message('note_on', note=lower_note[0], velocity=64, time=lower_note[1]))

                        # Đưa nốt cao vào Triangle (TR)
                        tr_track.append(Message('note_on', note=higher_note[0], velocity=64, time=higher_note[1]))

                        active_notes.clear()  # Reset danh sách nốt

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Xử lý tắt nốt trên cả 2 kênh
                    p1_track.append(Message('note_off', note=msg.note, velocity=0, time=msg.time))
                    tr_track.append(Message('note_off', note=msg.note, velocity=0, time=msg.time))

    new_midi.save(output_midi)
    print(f"Đã lưu file: {output_midi}")
app = Flask(__name__)
config_path = "config/default.yaml"
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
vocab_path = config['data']['vocab_path']
data_split = 'pickles/test_pieces.pkl'

ckpt_path = "musemorphose_pretrained_weights.pt"
# Khởi tạo mô hình và dữ liệu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = generate.model # Cấu hình mô hình
dset = generate.dset # Cấu hình dữ liệu
model.eval()
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
import fluidsynth
def midi_to_wav(midi_path, wav_path, soundfont_path='path_to_your_soundfont.sf2'):
    # Khởi tạo FluidSynth
    midi = pretty_midi.PrettyMIDI(midi_path)
    audio = midi.fluidsynth(fs=22050) # Sử dụng phương thức đúng `midi_to_audio`
    sf.write(wav_path, audio, 22050)
def midi_nes_to_wav(midi_path, wav_path, soundfont_path='path_to_your_soundfont.sf2'):
    # Khởi tạo FluidSynth
    midi = pretty_midi.PrettyMIDI(midi_path)
    audio = midi.fluidsynth(fs=22050,sf2_path="C:/Users/anhna\Downloads\8bitsf.SF2") # Sử dụng phương thức đúng `midi_to_audio`
    sf.write(wav_path, audio, 22050)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_music', methods=['POST'])
def generate_music():
    print(request.form)
    # Lấy tham số từ form
    polyphony_min = int((1-int(request.form['valence']))+int(request.form['energy'])+(1-int(request.form['danceability'])))
    polyphony_max = polyphony_min +2
    rhythmic_min = int((1-int(request.form['valence']))+int(request.form['energy'])+int(request.form['danceability']))
    rhythmic_max = rhythmic_min + 2
    pieces = random.sample(range(len(dset)), 1)
    print(polyphony_min,rhythmic_min)
    for p in pieces:
        # fetch test sample
        p_data = dset[p]
        p_data['enc_input'] = p_data['enc_input'][: p_data['enc_n_bars']]
        p_data['enc_padding_mask'] = p_data['enc_padding_mask'][: p_data['enc_n_bars']]
        orig_song = p_data['dec_input'].tolist()[:p_data['length']]
        orig_song = generate.word2event(orig_song, dset.idx2event)
        # output reference song's MIDI
        _, orig_tempo = remi2midi(orig_song, "original_music_input.mid", return_first_tempo=True, enforce_tempo=False)
        generate.generate_custom_music(model, dset, p_data, device, n_samples_per_piece=1, max_input_len=1280, temperature=1.2,
                              nucleus_p=0.9, polyphony_range=(polyphony_min, polyphony_max), rhythmic_freq_range=(rhythmic_min, rhythmic_max))
    output_midi_path = "generated_music_output.mid"
    # Lưu tệp MIDI tại thư mục dự án
    output_wav_path = "generated_music_output.wav"
    midi_to_wav("original_music_input.mid","original_music_input.wav")
    midi_to_wav(output_midi_path, output_wav_path)
    # Trả về đường dẫn WAV cho người dùng
    midi_to_wav('generated_music_NES.mid',"generated_music_NES.wav")
    split_piano_to_p1_tr_reverse("generated_music_output.mid","nes_music.mid")
    midi_nes_to_wav("nes_music.mid", "nes_music.wav")
    return render_template('result.html', generated_wav=output_wav_path, original_wav="original_music_input.wav")
@app.route('/play_generated_music')
def play_generated_music():
    # Trả về tệp WAV đã sinh ra để phát trực tiếp
    return send_file("generated_music_output.wav", as_attachment=False, mimetype="audio/wav")

@app.route('/play_original_music')
def play_original_music():
    # Trả về tệp WAV gốc để phát trực tiếp
    return send_file("original_music_input.wav", as_attachment=False, mimetype="audio/wav")
@app.route('/play_NES_music')
def play_NES_music():
    # Trả về tệp WAV gốc để phát trực tiếp
    return send_file("generated_music_NES.wav", as_attachment=False, mimetype="audio/wav")
@app.route('/play_NES_real_music')
def play_NES_real_music():
    # Trả về tệp WAV gốc để phát trực tiếp
    return send_file("nes_music.wav", as_attachment=False, mimetype="audio/wav")
@app.route('/download_generated_midi')
def download_generated_midi():
    # Trả về tệp MIDI đã sinh ra
    return send_file("generated_music_output.mid", as_attachment=True, mimetype="audio/midi")

@app.route('/download_original_midi')
def download_original_midi():
    # Trả về tệp MIDI gốc
    return send_file("original_music_input.mid", as_attachment=True, mimetype="audio/midi")

if __name__ == '__main__':
    app.run(debug=True)
