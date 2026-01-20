; ModuleID = '__compute_module_part_01'
source_filename = "__compute_module"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@broadcast_add_fusion.3.clone_parallel_bounds = private unnamed_addr constant [18 x [2 x [2 x i64]]] [[2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 0, i64 1], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 1, i64 2], [2 x i64] [i64 8330, i64 10000]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 0, i64 1666]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 1666, i64 3332]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 3332, i64 4998]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 4998, i64 6664]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 6664, i64 8330]], [2 x [2 x i64]] [[2 x i64] [i64 2, i64 3], [2 x i64] [i64 8330, i64 10000]]]

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define noalias noundef ptr @broadcast_add_fusion.3.clone(ptr readonly captures(none) %0) local_unnamed_addr #0 {
  %tid_gep = getelementptr inbounds nuw i8, ptr %0, i64 8
  %tids = load ptr, ptr %tid_gep, align 8
  %tid_x = load i64, ptr %tids, align 4
  %args_gep = getelementptr inbounds nuw i8, ptr %0, i64 24
  %args = load ptr, ptr %args_gep, align 8
  %arg1_gep = getelementptr i8, ptr %args, i64 16
  %arg1 = load ptr, ptr %arg1_gep, align 8, !invariant.load !1, !dereferenceable !2, !align !3
  %lo_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 0
  %up_dim_0_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i64 0, i64 %tid_x, i64 0, i64 1
  %lo_dim_0 = load i64, ptr %lo_dim_0_gep, align 16
  %up_dim_0 = load i64, ptr %up_dim_0_gep, align 8
  %lo_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 0
  %up_dim_1_gep = getelementptr inbounds [18 x [2 x [2 x i64]]], ptr @broadcast_add_fusion.3.clone_parallel_bounds, i64 0, i64 %tid_x, i64 1, i64 1
  %lo_dim_1 = load i64, ptr %lo_dim_1_gep, align 16
  %up_dim_1 = load i64, ptr %up_dim_1_gep, align 8
  %.not5 = icmp ult i64 %lo_dim_0, %up_dim_0
  br i1 %.not5, label %broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.lr.ph, label %return

broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.lr.ph: ; preds = %1
  %arg0 = load ptr, ptr %args, align 8, !invariant.load !1, !dereferenceable !4, !align !3
  %.not13 = icmp ult i64 %lo_dim_1, %up_dim_1
  %2 = load i32, ptr %arg0, align 64
  br i1 %.not13, label %broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.us, label %return

broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.us: ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.lr.ph, %broadcast_add_fusion.3.clone.loop_header.dim.1.broadcast_add_fusion.3.clone.loop_exit.dim.1_crit_edge.us
  %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us = phi i64 [ %invar.inc.us, %broadcast_add_fusion.3.clone.loop_header.dim.1.broadcast_add_fusion.3.clone.loop_exit.dim.1_crit_edge.us ], [ %lo_dim_0, %broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.lr.ph ]
  %3 = mul i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, 640000
  br label %broadcast_add_fusion.3.clone.loop_header.dim.2.preheader.us

broadcast_add_fusion.3.clone.loop_header.dim.2.preheader.us: ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.us, %broadcast_add_fusion.3.clone.loop_header.dim.2.preheader.us
  %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us = phi i64 [ %lo_dim_1, %broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.us ], [ %invar.inc3.us, %broadcast_add_fusion.3.clone.loop_header.dim.2.preheader.us ]
  %4 = shl i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, 6
  %5 = add i64 %4, %3
  %6 = lshr i64 %5, 32
  %7 = trunc nuw i64 %6 to i32
  %8 = add i32 %2, %7
  %9 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, i64 0
  %10 = insertelement <8 x i32> poison, i32 %8, i64 0
  %11 = shufflevector <8 x i32> %10, <8 x i32> poison, <8 x i32> zeroinitializer
  store <8 x i32> %11, ptr %9, align 64, !alias.scope !5
  %12 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, i64 8
  store <8 x i32> %11, ptr %12, align 32, !alias.scope !5
  %13 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, i64 16
  store <8 x i32> %11, ptr %13, align 64, !alias.scope !5
  %14 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, i64 24
  store <8 x i32> %11, ptr %14, align 32, !alias.scope !5
  %15 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, i64 32
  store <8 x i32> %11, ptr %15, align 64, !alias.scope !5
  %16 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, i64 40
  store <8 x i32> %11, ptr %16, align 32, !alias.scope !5
  %17 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, i64 48
  store <8 x i32> %11, ptr %17, align 64, !alias.scope !5
  %18 = getelementptr inbounds [3 x [10000 x [64 x i32]]], ptr %arg1, i64 0, i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, i64 56
  store <8 x i32> %11, ptr %18, align 32, !alias.scope !5
  %invar.inc3.us = add nuw nsw i64 %broadcast_add_fusion.3.clone.invar_address.dim.1.04.us, 1
  %exitcond.not = icmp eq i64 %invar.inc3.us, %up_dim_1
  br i1 %exitcond.not, label %broadcast_add_fusion.3.clone.loop_header.dim.1.broadcast_add_fusion.3.clone.loop_exit.dim.1_crit_edge.us, label %broadcast_add_fusion.3.clone.loop_header.dim.2.preheader.us, !llvm.loop !8

broadcast_add_fusion.3.clone.loop_header.dim.1.broadcast_add_fusion.3.clone.loop_exit.dim.1_crit_edge.us: ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.2.preheader.us
  %invar.inc.us = add nuw nsw i64 %broadcast_add_fusion.3.clone.invar_address.dim.0.06.us, 1
  %exitcond8.not = icmp eq i64 %invar.inc.us, %up_dim_0
  br i1 %exitcond8.not, label %return, label %broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.us, !llvm.loop !10

return:                                           ; preds = %broadcast_add_fusion.3.clone.loop_header.dim.1.broadcast_add_fusion.3.clone.loop_exit.dim.1_crit_edge.us, %broadcast_add_fusion.3.clone.loop_header.dim.1.preheader.lr.ph, %1
  ret ptr null
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="all" "prefer-vector-width"="256" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xla_dylib_index", i64 1}
!1 = !{}
!2 = !{i64 7680000}
!3 = !{i64 64}
!4 = !{i64 8}
!5 = !{!6}
!6 = !{!"result slice: {index:7, offset:15360064, size:7680000}", !7}
!7 = !{!"XLA host kernel broadcast_add_fusion.3.clone AA domain"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
!10 = distinct !{!10, !9}
