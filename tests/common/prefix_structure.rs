use asap_sketchlib::impl_fixed_matrix;
use asap_sketchlib::impl_hll_bucket_list;

impl_fixed_matrix!(FixMat1, i64, 3, 2048);
impl_fixed_matrix!(FixMat2, i64, 5, 2048);
impl_fixed_matrix!(FixMat3, i64, 7, 2048);
impl_fixed_matrix!(FixMat4, i64, 3, 4096);
impl_fixed_matrix!(FixMat5, i64, 5, 4096);
impl_fixed_matrix!(FixMat6, i64, 7, 4096);
impl_fixed_matrix!(FixMat7, i64, 3, 8192);
impl_fixed_matrix!(FixMat8, i64, 5, 8192);
impl_fixed_matrix!(FixMat9, i64, 7, 8192);
impl_fixed_matrix!(FixMat10, i64, 3, 16384);
impl_fixed_matrix!(FixMat11, i64, 5, 16384);
impl_fixed_matrix!(FixMat12, i64, 7, 16384);
impl_fixed_matrix!(FixMat13, i64, 3, 32768);
impl_fixed_matrix!(FixMat14, i64, 5, 32768);
impl_fixed_matrix!(FixMat15, i64, 7, 32768);

impl_hll_bucket_list!(HllRegP10, 10, 1_usize << 10);
impl_hll_bucket_list!(HllRegP11, 11, 1_usize << 11);
impl_hll_bucket_list!(HllRegP12, 12, 1_usize << 12);
impl_hll_bucket_list!(HllRegP13, 13, 1_usize << 13);
impl_hll_bucket_list!(HllRegP14, 14, 1_usize << 14);
impl_hll_bucket_list!(HllRegP15, 15, 1_usize << 15);
impl_hll_bucket_list!(HllRegP16, 16, 1_usize << 16);
impl_hll_bucket_list!(HllRegP17, 17, 1_usize << 17);
impl_hll_bucket_list!(HllRegP18, 18, 1_usize << 18);
