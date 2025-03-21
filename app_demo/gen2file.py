def generate_custom_music(model, dset, p_data, device, n_samples_per_piece=1, max_input_len=1280, temperature=1.2,
                          nucleus_p=0.9, polyphony_range=(-3, 4), rhythmic_freq_range=(-3, 4)):
    """
    Hàm sinh nhạc tùy chỉnh với polyphony và rhythmic frequency có thể tùy chỉnh.

    Parameters:
    - model: Mô hình MuseMorphose đã được huấn luyện.
    - dset: Bộ dữ liệu.
    - p_data: Dữ liệu bài nhạc mẫu.
    - device: Thiết bị (GPU hoặc CPU).
    - n_samples_per_piece: Số mẫu nhạc cần sinh (chỉ tạo 1 mẫu cho mỗi bài nhạc).
    - max_input_len: Chiều dài chuỗi đầu vào tối đa.
    - temperature: Tham số điều chỉnh độ "nóng" của sampling.
    - nucleus_p: Tham số xác định ngưỡng xác suất cho nucleus sampling.
    - polyphony_range: Phạm vi thay đổi cho lớp polyphony (theo khoảng [lower, upper]).
    - rhythmic_freq_range: Phạm vi thay đổi cho lớp rhythmic frequency (theo khoảng [lower, upper]).

    Returns:
    - Generated song with custom polyphony and rhythmic frequency adjustments.
    """

    # Lấy latent embedding nhanh cho bài nhạc
    p_latents = get_latent_embedding_fast(
        model, p_data,
        use_sampling=True,  # Sử dụng sampling cho việc sinh nhạc
        sampling_var=0.1  # Biến sampling, có thể điều chỉnh thêm
    )

    # Hàm tạo sự thay đổi ngẫu nhiên cho polyphony và rhythmic frequency
    def random_shift_attr_cls(n_samples, upper=4, lower=-3):
        return np.random.randint(lower, upper, (n_samples,))

    # Tạo sự thay đổi ngẫu nhiên cho polyphony và rhythmic frequency
    p_cls_diff = random_shift_attr_cls(n_samples_per_piece, *polyphony_range)
    r_cls_diff = random_shift_attr_cls(n_samples_per_piece, *rhythmic_freq_range)

    piece_entropies = []
    generated_songs = []

    # Chỉ tạo 1 mẫu nhạc cho mỗi bài
    for samp in range(n_samples_per_piece):
        # Tạo lớp polyphony và rhythmic frequency sau khi thay đổi
        p_polyph_cls = (p_data['polyph_cls_bar'] + p_cls_diff[samp]).clamp(0, 7).long()
        p_rfreq_cls = (p_data['rhymfreq_cls_bar'] + r_cls_diff[samp]).clamp(0, 7).long()

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
        remi2midi(song, out_file + '.mid', enforce_tempo=True)

        # Lưu siêu dữ liệu về nhạc đã sinh
        np.save(out_file + '-POLYCLS.npy', tensor_to_numpy(p_polyph_cls))
        np.save(out_file + '-RHYMCLS.npy', tensor_to_numpy(p_rfreq_cls))

        print(f"[info] Entropy: {entropies.mean():.4f} (+/- {entropies.std():.4f})")

    # Thống kê tổng hợp
    print(f"[time stats] Generation time: {np.mean(piece_entropies):.2f} secs (+/- {np.std(piece_entropies):.2f})")

    return generated_songs