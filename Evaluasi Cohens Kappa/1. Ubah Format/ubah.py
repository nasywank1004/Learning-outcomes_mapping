import pandas as pd
import re
import os

# --- 1. KONFIGURASI NAMA FILE, SHEET, DAN KOLOM ---
# Masukkan daftar nama file Anda di sini
daftar_file = [
    'Anotasi Skill - A.xlsx', 
    'Anotasi Skill - B.xlsx', 
    'Anotasi Skill - C.xlsx'
]

# Nama sheet yang akan diproses
target_sheets = ['CLO', 'SFIA']

# Nama kolom yang akan diubah isinya
target_columns = ['Hasil Ekstraksi Sistem', 'Hasil Anotasi Pakar']

# -------------------------------------------------

def bersihkan_dan_format(teks):
    """Fungsi untuk menghapus simbol list dan mengganti koma dengan baris baru"""
    if pd.isna(teks) or teks == "":
        return teks
    
    # Menghapus karakter: [ ] dan '
    # Menggunakan regex agar lebih bersih
    bersih = re.sub(r"[\[\]']", "", str(teks))
    
    # Mengganti koma (dengan atau tanpa spasi) menjadi newline (\n)
    # Ini yang membuat teks bertumpuk ke bawah di dalam satu sel
    hasil = bersih.replace(", ", "\n").replace(",", "\n").strip()
    return hasil

def main():
    print("--- MEMULAI PROSES KONVERSI ---")
    
    for nama_file in daftar_file:
        if not os.path.exists(nama_file):
            print(f"⚠️ Peringatan: File '{nama_file}' tidak ditemukan. Dilewati.")
            continue
            
        print(f"📂 Memproses File: {nama_file}...")
        
        # Membaca seluruh sheet (sheet_name=None mengembalikan dictionary)
        dict_df = pd.read_excel(nama_file, sheet_name=None)
        
        nama_output = f"HASIL_{nama_file}"
        
        # Menggunakan ExcelWriter untuk menyimpan banyak sheet sekaligus
        with pd.ExcelWriter(nama_output, engine='openpyxl') as writer:
            for sheet_name, df in dict_df.items():
                
                # Cek apakah sheet ini termasuk yang ingin diproses
                if sheet_name in target_sheets:
                    print(f"   -> Mengolah Sheet: {sheet_name}")
                    
                    for col in target_columns:
                        if col in df.columns:
                            print(f"      -> Membersihkan Kolom: {col}")
                            df[col] = df[col].apply(bersihkan_dan_format)
                
                # Tulis data ke sheet (baik yang diubah maupun yang tidak)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Selesai! File disimpan sebagai: {nama_output}\n")

    print("--- SEMUA PROSES BERHASIL DISELESAIKAN ---")

if __name__ == "__main__":
    main()