import Prng
open Prng
set_option maxRecDepth 4000
set_option maxHeartbeats 4000000

theorem fnv_0 : fnv "" = 14695981039346656037 := by decide +kernel
theorem fnv_1 : fnv "a" = 12638187200555641996 := by decide +kernel
theorem fnv_2 : fnv "ab" = 620445648566982762 := by decide +kernel
theorem fnv_3 : fnv "abc" = 16654208175385433931 := by decide +kernel
theorem fnv_4 : fnv "kick" = 17268634781200901759 := by decide +kernel
theorem fnv_5 : fnv "snare" = 12636623752503768950 := by decide +kernel
theorem fnv_6 : fnv "hat" = 3699733590342344130 := by decide +kernel
theorem fnv_7 : fnv "clap" = 1411348042330209891 := by decide +kernel
theorem fnv_8 : fnv "sample|kick" = 16329764846204387787 := by decide +kernel
theorem fnv_9 : fnv "sample|kick-click" = 3161087193363804242 := by decide +kernel
theorem fnv_10 : fnv "sample|snare" = 17540238934235548858 := by decide +kernel
theorem fnv_11 : fnv "sample|hat" = 13156567268615936062 := by decide +kernel
theorem fnv_12 : fnv "sample|clap" = 4822278063392861799 := by decide +kernel
theorem fnv_13 : fnv "0|1" = 5706035542692955816 := by decide +kernel
theorem fnv_14 : fnv "hum|123" = 2293812879092750505 := by decide +kernel
theorem fnv_15 : fnv "kick|1" = 11884827862455425272 := by decide +kernel
theorem fnv_16 : fnv "prob|0" = 10878282062238399698 := by decide +kernel
theorem fnv_17 : fnv "x|18446744073709551615" = 7408434059189402096 := by decide +kernel
theorem fnv_18 : fnv "é" = 775207407765167617 := by decide +kernel
theorem flt_19_seed : maskSeed 1 = 1 := by decide +kernel
theorem flt_19_key : fltKey 1 [.str "kick", .str "hum", .int 0] = 9009181385810910116 := by decide +kernel
theorem flt_19_out : splitmix64 9009181385810910116 = 12757181382089057799 := by decide +kernel
theorem flt_19_bits : fltBitsOfOut 12757181382089057799 = 0x3fe621539c0cd4a5 := by decide +kernel
theorem flt_20_seed : maskSeed 1 = 1 := by decide +kernel
theorem flt_20_key : fltKey 1 [.str "kick", .str "hum", .int 1] = 15900448910310693321 := by decide +kernel
theorem flt_20_out : splitmix64 15900448910310693321 = 467639867812890905 := by decide +kernel
theorem flt_20_bits : fltBitsOfOut 467639867812890905 = 0x3f99f58fedaf3124 := by decide +kernel
theorem flt_21_seed : maskSeed 1 = 1 := by decide +kernel
theorem flt_21_key : fltKey 1 [.str "kick", .str "prob", .int 0] = 8663828149854812398 := by decide +kernel
theorem flt_21_out : splitmix64 8663828149854812398 = 16451063564222362334 := by decide +kernel
theorem flt_21_bits : fltBitsOfOut 16451063564222362334 = 0x3fec89bd64ce5b0e := by decide +kernel
theorem flt_22_seed : maskSeed 1 = 1 := by decide +kernel
theorem flt_22_key : fltKey 1 [.str "hat", .str "prob", .int 31] = 17235206141895795746 := by decide +kernel
theorem flt_22_out : splitmix64 17235206141895795746 = 11243530964749613534 := by decide +kernel
theorem flt_22_bits : fltBitsOfOut 11243530964749613534 = 0x3fe381217aeee02f := by decide +kernel
theorem flt_23_seed : maskSeed 1 = 1 := by decide +kernel
theorem flt_23_key : fltKey 1 [.str "snare", .str "hum", .int 7] = 16875273023499940664 := by decide +kernel
theorem flt_23_out : splitmix64 16875273023499940664 = 17889463194238400601 := by decide +kernel
theorem flt_23_bits : fltBitsOfOut 17889463194238400601 = 0x3fef08847fc46027 := by decide +kernel
theorem flt_24_seed : maskSeed 1 = 1 := by decide +kernel
theorem flt_24_key : fltKey 1 [.int 0] = 5706035542692955816 := by decide +kernel
theorem flt_24_out : splitmix64 5706035542692955816 = 3329825183073691434 := by decide +kernel
theorem flt_24_bits : fltBitsOfOut 3329825183073691434 = 0x3fc71af52e50a0f4 := by decide +kernel
theorem flt_25_seed : maskSeed 1 = 1 := by decide +kernel
theorem flt_25_key : fltKey 1 [.int 1] = 4928820459719463363 := by decide +kernel
theorem flt_25_out : splitmix64 4928820459719463363 = 10418166507254056483 := by decide +kernel
theorem flt_25_bits : fltBitsOfOut 10418166507254056483 = 0x3fe2129867b27801 := by decide +kernel
theorem flt_26_seed : maskSeed 1 = 1 := by decide +kernel
theorem flt_26_key : fltKey 1 [.int 41] = 4159465512594866461 := by decide +kernel
theorem flt_26_out : splitmix64 4159465512594866461 = 4262026829083468677 := by decide +kernel
theorem flt_26_bits : fltBitsOfOut 4262026829083468677 = 0x3fcd92e168f15a32 := by decide +kernel
theorem flt_27_seed : maskSeed 2 = 2 := by decide +kernel
theorem flt_27_key : fltKey 2 [.str "kick", .str "hum", .int 0] = 2773414985869355725 := by decide +kernel
theorem flt_27_out : splitmix64 2773414985869355725 = 6255470360344140550 := by decide +kernel
theorem flt_27_bits : fltBitsOfOut 6255470360344140550 = 0x3fd5b3f94996d307 := by decide +kernel
theorem flt_28_seed : maskSeed 2 = 2 := by decide +kernel
theorem flt_28_key : fltKey 2 [.str "kick", .str "hum", .int 1] = 12277219026948343142 := by decide +kernel
theorem flt_28_out : splitmix64 12277219026948343142 = 4406150150260045110 := by decide +kernel
theorem flt_28_bits : fltBitsOfOut 4406150150260045110 = 0x3fce92e51bc2a45b := by decide +kernel
theorem flt_29_seed : maskSeed 2 = 2 := by decide +kernel
theorem flt_29_key : fltKey 2 [.str "kick", .str "prob", .int 0] = 5826631512820790366 := by decide +kernel
theorem flt_29_out : splitmix64 5826631512820790366 = 16393747132694783303 := by decide +kernel
theorem flt_29_bits : fltBitsOfOut 16393747132694783303 = 0x3fec70494519bb3d := by decide +kernel
theorem flt_30_seed : maskSeed 2 = 2 := by decide +kernel
theorem flt_30_key : fltKey 2 [.str "hat", .str "prob", .int 31] = 15030520005233326910 := by decide +kernel
theorem flt_30_out : splitmix64 15030520005233326910 = 12713720907164796623 := by decide +kernel
theorem flt_30_bits : fltBitsOfOut 12713720907164796623 = 0x3fe60e06b9c1a079 := by decide +kernel
theorem flt_31_seed : maskSeed 2 = 2 := by decide +kernel
theorem flt_31_key : fltKey 2 [.str "snare", .str "hum", .int 7] = 2518561563556561199 := by decide +kernel
theorem flt_31_out : splitmix64 2518561563556561199 = 663972912476720797 := by decide +kernel
theorem flt_31_bits : fltBitsOfOut 663972912476720797 = 0x3fa26dcfb129501d := by decide +kernel
theorem flt_32_seed : maskSeed 2 = 2 := by decide +kernel
theorem flt_32_key : fltKey 2 [.int 0] = 5706038841227840449 := by decide +kernel
theorem flt_32_out : splitmix64 5706038841227840449 = 2339962062033943810 := by decide +kernel
theorem flt_32_bits : fltBitsOfOut 2339962062033943810 = 0x3fc03c9b8c83b63f := by decide +kernel
theorem flt_33_seed : maskSeed 2 = 2 := by decide +kernel
theorem flt_33_key : fltKey 2 [.int 1] = 4928821559231091574 := by decide +kernel
theorem flt_33_out : splitmix64 4928821559231091574 = 5770775617285411770 := by decide +kernel
theorem flt_33_bits : fltBitsOfOut 5770775617285411770 = 0x3fd4057a7656b215 := by decide +kernel
theorem flt_34_seed : maskSeed 2 = 2 := by decide +kernel
theorem flt_34_key : fltKey 2 [.int 41] = 4159462214059981828 := by decide +kernel
theorem flt_34_out : splitmix64 4159462214059981828 = 3390881933790995449 := by decide +kernel
theorem flt_34_bits : fltBitsOfOut 3390881933790995449 = 0x3fc7876a93997372 := by decide +kernel
theorem flt_35_seed : maskSeed 7 = 7 := by decide +kernel
theorem flt_35_key : fltKey 7 [.str "kick", .str "hum", .int 0] = 15129922893665966410 := by decide +kernel
theorem flt_35_out : splitmix64 15129922893665966410 = 4086868642287052498 := by decide +kernel
theorem flt_35_bits : fltBitsOfOut 4086868642287052498 = 0x3fcc5bbcb04199f3 := by decide +kernel
theorem flt_36_seed : maskSeed 7 = 7 := by decide +kernel
theorem flt_36_key : fltKey 7 [.str "kick", .str "hum", .int 1] = 10577207867619247991 := by decide +kernel
theorem flt_36_out : splitmix64 10577207867619247991 = 18105136213321633839 := by decide +kernel
theorem flt_36_bits : fltBitsOfOut 18105136213321633839 = 0x3fef684baebfa31a := by decide +kernel
theorem flt_37_seed : maskSeed 7 = 7 := by decide +kernel
theorem flt_37_key : fltKey 7 [.str "kick", .str "prob", .int 0] = 14635136115658558815 := by decide +kernel
theorem flt_37_out : splitmix64 14635136115658558815 = 5510905795640608826 := by decide +kernel
theorem flt_37_bits : fltBitsOfOut 5510905795640608826 = 0x3fd31eaae7e0f332 := by decide +kernel
theorem flt_38_seed : maskSeed 7 = 7 := by decide +kernel
theorem flt_38_key : fltKey 7 [.str "hat", .str "prob", .int 31] = 18366438950068604950 := by decide +kernel
theorem flt_38_out : splitmix64 18366438950068604950 = 8556394189718001347 := by decide +kernel
theorem flt_38_bits : fltBitsOfOut 8556394189718001347 = 0x3fddaf9acba316e6 := by decide +kernel
theorem flt_39_seed : maskSeed 7 = 7 := by decide +kernel
theorem flt_39_key : fltKey 7 [.str "snare", .str "hum", .int 7] = 311037092872690550 := by decide +kernel
theorem flt_39_out : splitmix64 311037092872690550 = 9965958867797129579 := by decide +kernel
theorem flt_39_bits : fltBitsOfOut 9965958867797129579 = 0x3fe149c6593a2677 := by decide +kernel
theorem flt_40_seed : maskSeed 7 = 7 := by decide +kernel
theorem flt_40_key : fltKey 7 [.int 0] = 5706042139762725082 := by decide +kernel
theorem flt_40_out : splitmix64 5706042139762725082 = 5373665931711646291 := by decide +kernel
theorem flt_40_bits : fltBitsOfOut 5373665931711646291 = 0x3fd2a4c62c6a52a3 := by decide +kernel
theorem flt_41_seed : maskSeed 7 = 7 := by decide +kernel
theorem flt_41_key : fltKey 7 [.int 1] = 4928827056789232629 := by decide +kernel
theorem flt_41_out : splitmix64 4928827056789232629 = 12462952158586089905 := by decide +kernel
theorem flt_41_bits : fltBitsOfOut 12462952158586089905 = 0x3fe59ea99e9d26c6 := by decide +kernel
theorem flt_42_seed : maskSeed 7 = 7 := by decide +kernel
theorem flt_42_key : fltKey 7 [.int 41] = 4159458915525097195 := by decide +kernel
theorem flt_42_out : splitmix64 4159458915525097195 = 12892727376979303825 := by decide +kernel
theorem flt_42_bits : fltBitsOfOut 12892727376979303825 = 0x3fe65d8567b459c4 := by decide +kernel
theorem flt_43_seed : maskSeed 13 = 13 := by decide +kernel
theorem flt_43_key : fltKey 13 [.str "kick", .str "hum", .int 0] = 1481068505140141070 := by decide +kernel
theorem flt_43_out : splitmix64 1481068505140141070 = 6058400220100580244 := by decide +kernel
theorem flt_43_bits : fltBitsOfOut 6058400220100580244 = 0x3fd504f0b9b08f95 := by decide +kernel
theorem flt_44_seed : maskSeed 13 = 13 := by decide +kernel
theorem flt_44_key : fltKey 13 [.str "kick", .str "hum", .int 1] = 6376447196133481685 := by decide +kernel
theorem flt_44_out : splitmix64 6376447196133481685 = 2766249899105893150 := by decide +kernel
theorem flt_44_bits : fltBitsOfOut 2766249899105893150 = 0x3fc331d8d042155e := by decide +kernel
theorem flt_45_seed : maskSeed 13 = 13 := by decide +kernel
theorem flt_45_key : fltKey 13 [.str "kick", .str "prob", .int 0] = 6113546062030653267 := by decide +kernel
theorem flt_45_out : splitmix64 6113546062030653267 = 16741192582307531419 := by decide +kernel
theorem flt_45_bits : fltBitsOfOut 16741192582307531419 = 0x3fed0a953e8f1b09 := by decide +kernel
theorem flt_46_seed : maskSeed 13 = 13 := by decide +kernel
theorem flt_46_key : fltKey 13 [.str "hat", .str "prob", .int 31] = 6569928357206616315 := by decide +kernel
theorem flt_46_out : splitmix64 6569928357206616315 = 10070074555902414558 := by decide +kernel
theorem flt_46_bits : fltBitsOfOut 10070074555902414558 = 0x3fe17802ee8fc86a := by decide +kernel
theorem flt_47_seed : maskSeed 13 = 13 := by decide +kernel
theorem flt_47_key : fltKey 13 [.str "snare", .str "hum", .int 7] = 11592443830190405426 := by decide +kernel
theorem flt_47_out : splitmix64 11592443830190405426 = 1985904175011501217 := by decide +kernel
theorem flt_47_bits : fltBitsOfOut 1985904175011501217 = 0x3fbb8f59534d8695 := by decide +kernel
theorem flt_48_seed : maskSeed 13 = 13 := by decide +kernel
theorem flt_48_key : fltKey 13 [.int 0] = 9651124919196039521 := by decide +kernel
theorem flt_48_out : splitmix64 9651124919196039521 = 17623461022340051418 := by decide +kernel
theorem flt_48_bits : fltBitsOfOut 17623461022340051418 = 0x3fee92638da7b0ae := by decide +kernel
theorem flt_49_seed : maskSeed 13 = 13 := by decide +kernel
theorem flt_49_key : fltKey 13 [.int 1] = 8939408794537622224 := by decide +kernel
theorem flt_49_out : splitmix64 8939408794537622224 = 18295920571732716982 := by decide +kernel
theorem flt_49_bits : fltBitsOfOut 18295920571732716982 = 0x3fefbd055a5e9c7e := by decide +kernel
theorem flt_50_seed : maskSeed 13 = 13 := by decide +kernel
theorem flt_50_key : fltKey 13 [.int 41] = 14593306531588440362 := by decide +kernel
theorem flt_50_out : splitmix64 14593306531588440362 = 16361030053194388779 := by decide +kernel
theorem flt_50_bits : fltBitsOfOut 16361030053194388779 = 0x3fec61c1c4c4048b := by decide +kernel
theorem flt_51_seed : maskSeed 41 = 41 := by decide +kernel
theorem flt_51_key : fltKey 41 [.str "kick", .str "hum", .int 0] = 13391642200187286332 := by decide +kernel
theorem flt_51_out : splitmix64 13391642200187286332 = 6713537841450575239 := by decide +kernel
theorem flt_51_bits : fltBitsOfOut 6713537841450575239 = 0x3fd74ad1c6347379 := by decide +kernel
theorem flt_52_seed : maskSeed 41 = 41 := by decide +kernel
theorem flt_52_key : fltKey 41 [.str "kick", .str "hum", .int 1] = 17199544154297134031 := by decide +kernel
theorem flt_52_out : splitmix64 17199544154297134031 = 4936374817134311743 := by decide +kernel
theorem flt_52_bits : fltBitsOfOut 4936374817134311743 = 0x3fd12061af7f5e5e := by decide +kernel
theorem flt_53_seed : maskSeed 41 = 41 := by decide +kernel
theorem flt_53_key : fltKey 41 [.str "kick", .str "prob", .int 0] = 8939068366233085007 := by decide +kernel
theorem flt_53_out : splitmix64 8939068366233085007 = 9823027550008932344 := by decide +kernel
theorem flt_53_bits : fltBitsOfOut 9823027550008932344 = 0x3fe10a4cf063250f := by decide +kernel
theorem flt_54_seed : maskSeed 41 = 41 := by decide +kernel
theorem flt_54_key : fltKey 41 [.str "hat", .str "prob", .int 31] = 18297117650938998764 := by decide +kernel
theorem flt_54_out : splitmix64 18297117650938998764 = 8226272594980673756 := by decide +kernel
theorem flt_54_bits : fltBitsOfOut 8226272594980673756 = 0x3fdc8a65d5ca56a6 := by decide +kernel
theorem flt_55_seed : maskSeed 41 = 41 := by decide +kernel
theorem flt_55_key : fltKey 41 [.str "snare", .str "hum", .int 7] = 7120598205142924916 := by decide +kernel
theorem flt_55_out : splitmix64 7120598205142924916 = 8413439981506867135 := by decide +kernel
theorem flt_55_bits : fltBitsOfOut 8413439981506867135 = 0x3fdd30a2c58f324c := by decide +kernel
theorem flt_56_seed : maskSeed 41 = 41 := by decide +kernel
theorem flt_56_key : fltKey 41 [.int 0] = 9655903396731298402 := by decide +kernel
theorem flt_56_out : splitmix64 9655903396731298402 = 2515536648523032331 := by decide +kernel
theorem flt_56_bits : fltBitsOfOut 2515536648523032331 = 0x3fc1747da0815894 := by decide +kernel
theorem flt_57_seed : maskSeed 41 = 41 := by decide +kernel
theorem flt_57_key : fltKey 41 [.int 1] = 8942247733561095351 := by decide +kernel
theorem flt_57_out : splitmix64 8942247733561095351 = 18364650057183304505 := by decide +kernel
theorem flt_57_bits : fltBitsOfOut 18364650057183304505 = 0x3fefdb8afda95893 := by decide +kernel
theorem flt_58_seed : maskSeed 41 = 41 := by decide +kernel
theorem flt_58_key : fltKey 41 [.int 41] = 14588523656006668637 := by decide +kernel
theorem flt_58_out : splitmix64 14588523656006668637 = 2284487120611317957 := by decide +kernel
theorem flt_58_bits : fltBitsOfOut 2284487120611317957 = 0x3fbfb420eeb5fe19 := by decide +kernel
theorem flt_59_seed : maskSeed (-1 : Int) = 18446744073709551615 := by decide +kernel
theorem flt_59_key : fltKey 18446744073709551615 [.str "kick", .str "hum", .int 0] = 9281706059039921865 := by decide +kernel
theorem flt_59_out : splitmix64 9281706059039921865 = 12846207557701622176 := by decide +kernel
theorem flt_59_bits : fltBitsOfOut 12846207557701622176 = 0x3fe648dcb6c577cb := by decide +kernel
theorem flt_60_seed : maskSeed (-1 : Int) = 18446744073709551615 := by decide +kernel
theorem flt_60_key : fltKey 18446744073709551615 [.str "kick", .str "hum", .int 1] = 5696082070875002596 := by decide +kernel
theorem flt_60_out : splitmix64 5696082070875002596 = 6411241653308588440 := by decide +kernel
theorem flt_60_bits : fltBitsOfOut 6411241653308588440 = 0x3fd63e539430a960 := by decide +kernel
theorem flt_61_seed : maskSeed (-1 : Int) = 18446744073709551615 := by decide +kernel
theorem flt_61_key : fltKey 18446744073709551615 [.str "kick", .str "prob", .int 0] = 5128112236307651194 := by decide +kernel
theorem flt_61_out : splitmix64 5128112236307651194 = 10573257400230596023 := by decide +kernel
theorem flt_61_bits : fltBitsOfOut 10573257400230596023 = 0x3fe2577832c0707a := by decide +kernel
theorem flt_62_seed : maskSeed (-1 : Int) = 18446744073709551615 := by decide +kernel
theorem flt_62_key : fltKey 18446744073709551615 [.str "hat", .str "prob", .int 31] = 2264120264356947807 := by decide +kernel
theorem flt_62_out : splitmix64 2264120264356947807 = 475549350226305700 := by decide +kernel
theorem flt_62_bits : fltBitsOfOut 475549350226305700 = 0x3f9a65f67535a08b := by decide +kernel
theorem flt_63_seed : maskSeed (-1 : Int) = 18446744073709551615 := by decide +kernel
theorem flt_63_key : fltKey 18446744073709551615 [.str "snare", .str "hum", .int 7] = 2589979805664392937 := by decide +kernel
theorem flt_63_out : splitmix64 2589979805664392937 = 5931852765154887047 := by decide +kernel
theorem flt_63_bits : fltBitsOfOut 5931852765154887047 = 0x3fd4948b2a57be71 := by decide +kernel
theorem flt_64_seed : maskSeed (-1 : Int) = 18446744073709551615 := by decide +kernel
theorem flt_64_key : fltKey 18446744073709551615 [.int 0] = 17975476482421440344 := by decide +kernel
theorem flt_64_out : splitmix64 17975476482421440344 = 120161086863855059 := by decide +kernel
theorem flt_64_bits : fltBitsOfOut 120161086863855059 = 0x3f7aae5df32584dd := by decide +kernel
theorem flt_65_seed : maskSeed (-1 : Int) = 18446744073709551615 := by decide +kernel
theorem flt_65_key : fltKey 18446744073709551615 [.int 1] = 10206727706563192369 := by decide +kernel
theorem flt_65_out : splitmix64 10206727706563192369 = 4043171810137051012 := by decide +kernel
theorem flt_65_bits : fltBitsOfOut 4043171810137051012 = 0x3fcc0e1dab7a8bc8 := by decide +kernel
theorem flt_66_seed : maskSeed (-1 : Int) = 18446744073709551615 := by decide +kernel
theorem flt_66_key : fltKey 18446744073709551615 [.int 41] = 14595955610332575703 := by decide +kernel
theorem flt_66_out : splitmix64 14595955610332575703 = 7560668312444837038 := by decide +kernel
theorem flt_66_bits : fltBitsOfOut 7560668312444837038 = 0x3fda3b38f168f4e7 := by decide +kernel
theorem flt_67_seed : maskSeed 18446744073709551615 = 18446744073709551615 := by decide +kernel
theorem flt_67_key : fltKey 18446744073709551615 [.str "kick", .str "hum", .int 0] = 9281706059039921865 := by decide +kernel
theorem flt_67_out : splitmix64 9281706059039921865 = 12846207557701622176 := by decide +kernel
theorem flt_67_bits : fltBitsOfOut 12846207557701622176 = 0x3fe648dcb6c577cb := by decide +kernel
theorem flt_68_seed : maskSeed 18446744073709551615 = 18446744073709551615 := by decide +kernel
theorem flt_68_key : fltKey 18446744073709551615 [.str "kick", .str "hum", .int 1] = 5696082070875002596 := by decide +kernel
theorem flt_68_out : splitmix64 5696082070875002596 = 6411241653308588440 := by decide +kernel
theorem flt_68_bits : fltBitsOfOut 6411241653308588440 = 0x3fd63e539430a960 := by decide +kernel
theorem flt_69_seed : maskSeed 18446744073709551615 = 18446744073709551615 := by decide +kernel
theorem flt_69_key : fltKey 18446744073709551615 [.str "kick", .str "prob", .int 0] = 5128112236307651194 := by decide +kernel
theorem flt_69_out : splitmix64 5128112236307651194 = 10573257400230596023 := by decide +kernel
theorem flt_69_bits : fltBitsOfOut 10573257400230596023 = 0x3fe2577832c0707a := by decide +kernel
theorem flt_70_seed : maskSeed 18446744073709551615 = 18446744073709551615 := by decide +kernel
theorem flt_70_key : fltKey 18446744073709551615 [.str "hat", .str "prob", .int 31] = 2264120264356947807 := by decide +kernel
theorem flt_70_out : splitmix64 2264120264356947807 = 475549350226305700 := by decide +kernel
theorem flt_70_bits : fltBitsOfOut 475549350226305700 = 0x3f9a65f67535a08b := by decide +kernel
theorem flt_71_seed : maskSeed 18446744073709551615 = 18446744073709551615 := by decide +kernel
theorem flt_71_key : fltKey 18446744073709551615 [.str "snare", .str "hum", .int 7] = 2589979805664392937 := by decide +kernel
theorem flt_71_out : splitmix64 2589979805664392937 = 5931852765154887047 := by decide +kernel
theorem flt_71_bits : fltBitsOfOut 5931852765154887047 = 0x3fd4948b2a57be71 := by decide +kernel
theorem flt_72_seed : maskSeed 18446744073709551615 = 18446744073709551615 := by decide +kernel
theorem flt_72_key : fltKey 18446744073709551615 [.int 0] = 17975476482421440344 := by decide +kernel
theorem flt_72_out : splitmix64 17975476482421440344 = 120161086863855059 := by decide +kernel
theorem flt_72_bits : fltBitsOfOut 120161086863855059 = 0x3f7aae5df32584dd := by decide +kernel
theorem flt_73_seed : maskSeed 18446744073709551615 = 18446744073709551615 := by decide +kernel
theorem flt_73_key : fltKey 18446744073709551615 [.int 1] = 10206727706563192369 := by decide +kernel
theorem flt_73_out : splitmix64 10206727706563192369 = 4043171810137051012 := by decide +kernel
theorem flt_73_bits : fltBitsOfOut 4043171810137051012 = 0x3fcc0e1dab7a8bc8 := by decide +kernel
theorem flt_74_seed : maskSeed 18446744073709551615 = 18446744073709551615 := by decide +kernel
theorem flt_74_key : fltKey 18446744073709551615 [.int 41] = 14595955610332575703 := by decide +kernel
theorem flt_74_out : splitmix64 14595955610332575703 = 7560668312444837038 := by decide +kernel
theorem flt_74_bits : fltBitsOfOut 7560668312444837038 = 0x3fda3b38f168f4e7 := by decide +kernel
theorem flt_75_seed : maskSeed 12345678901234567890 = 12345678901234567890 := by decide +kernel
theorem flt_75_key : fltKey 12345678901234567890 [.str "kick", .str "hum", .int 0] = 6410491097820105274 := by decide +kernel
theorem flt_75_out : splitmix64 6410491097820105274 = 8238595112922386140 := by decide +kernel
theorem flt_75_bits : fltBitsOfOut 8238595112922386140 = 0x3fdc9557a6bc8084 := by decide +kernel
theorem flt_76_seed : maskSeed 12345678901234567890 = 12345678901234567890 := by decide +kernel
theorem flt_76_key : fltKey 12345678901234567890 [.str "kick", .str "hum", .int 1] = 15926144789950209379 := by decide +kernel
theorem flt_76_out : splitmix64 15926144789950209379 = 4863424868985589985 := by decide +kernel
theorem flt_76_bits : fltBitsOfOut 4863424868985589985 = 0x3fd0df96c9a6ebf4 := by decide +kernel
theorem flt_77_seed : maskSeed 12345678901234567890 = 12345678901234567890 := by decide +kernel
theorem flt_77_key : fltKey 12345678901234567890 [.str "kick", .str "prob", .int 0] = 12795573331833631869 := by decide +kernel
theorem flt_77_out : splitmix64 12795573331833631869 = 5514382935022184792 := by decide +kernel
theorem flt_77_bits : fltBitsOfOut 5514382935022184792 = 0x3fd321c184075e4a := by decide +kernel
theorem flt_78_seed : maskSeed 12345678901234567890 = 12345678901234567890 := by decide +kernel
theorem flt_78_key : fltKey 12345678901234567890 [.str "hat", .str "prob", .int 31] = 15901382679292349963 := by decide +kernel
theorem flt_78_out : splitmix64 15901382679292349963 = 13910272450057628076 := by decide +kernel
theorem flt_78_bits : fltBitsOfOut 13910272450057628076 = 0x3fe82166e2fe52cd := by decide +kernel
theorem flt_79_seed : maskSeed 12345678901234567890 = 12345678901234567890 := by decide +kernel
theorem flt_79_key : fltKey 12345678901234567890 [.str "snare", .str "hum", .int 7] = 3619457969207445282 := by decide +kernel
theorem flt_79_out : splitmix64 3619457969207445282 = 14142616848627420883 := by decide +kernel
theorem flt_79_bits : fltBitsOfOut 14142616848627420883 = 0x3fe8889562fa1c8a := by decide +kernel
theorem flt_80_seed : maskSeed 12345678901234567890 = 12345678901234567890 := by decide +kernel
theorem flt_80_key : fltKey 12345678901234567890 [.int 0] = 9067943466503456881 := by decide +kernel
theorem flt_80_out : splitmix64 9067943466503456881 = 4458133278213669396 := by decide +kernel
theorem flt_80_bits : fltBitsOfOut 4458133278213669396 = 0x3fceef3c4c54a301 := by decide +kernel
theorem flt_81_seed : maskSeed 12345678901234567890 = 12345678901234567890 := by decide +kernel
theorem flt_81_key : fltKey 12345678901234567890 [.int 1] = 16367454197549552104 := by decide +kernel
theorem flt_81_out : splitmix64 16367454197549552104 = 11035409188825569172 := by decide +kernel
theorem flt_81_bits : fltBitsOfOut 11035409188825569172 = 0x3fe324b4c60d60f3 := by decide +kernel
theorem flt_82_seed : maskSeed 12345678901234567890 = 12345678901234567890 := by decide +kernel
theorem flt_82_key : fltKey 12345678901234567890 [.int 41] = 15624724306886143398 := by decide +kernel
theorem flt_82_out : splitmix64 15624724306886143398 = 4708991831262369427 := by decide +kernel
theorem flt_82_bits : fltBitsOfOut 4708991831262369427 = 0x3fd0566cc7c22706 := by decide +kernel
theorem noise_83_0_key : noiseKey "kick-click" 0 = 667070431642405526 := by decide +kernel
theorem noise_83_0_out : splitmix64 667070431642405526 = 3345034419766308993 := by decide +kernel
theorem noise_83_0_bits : noiseBitsOfOut 3345034419766308993 = 0xbfe465033ac017f4 := by decide +kernel
theorem noise_83_1_key : noiseKey "kick-click" 1 = 7268749302957334673 := by decide +kernel
theorem noise_83_1_out : splitmix64 7268749302957334673 = 4923864183436952601 := by decide +kernel
theorem noise_83_1_bits : noiseBitsOfOut 4923864183436952601 = 0xbfddd575ce73fc5e := by decide +kernel
theorem noise_83_2_key : noiseKey "kick-click" 2 = 4869407824214494952 := by decide +kernel
theorem noise_83_2_out : splitmix64 4869407824214494952 = 8253900080084796457 := by decide +kernel
theorem noise_83_2_bits : noiseBitsOfOut 8253900080084796457 = 0xbfbae8833765f478 := by decide +kernel
theorem noise_83_3_key : noiseKey "kick-click" 3 = 17723874476856155427 := by decide +kernel
theorem noise_83_3_out : splitmix64 17723874476856155427 = 15368423811760803857 := by decide +kernel
theorem noise_83_3_bits : noiseBitsOfOut 15368423811760803857 = 0x3fe551e702026a14 := by decide +kernel
theorem noise_83_4_key : noiseKey "kick-click" 4 = 4769307296494470074 := by decide +kernel
theorem noise_83_4_out : splitmix64 4769307296494470074 = 4782522591842732263 := by decide +kernel
theorem noise_83_4_bits : noiseBitsOfOut 4782522591842732263 = 0xbfded0888553777c := by decide +kernel
theorem noise_83_5_key : noiseKey "kick-click" 5 = 4460984222111560245 := by decide +kernel
theorem noise_83_5_out : splitmix64 4460984222111560245 = 13277290101518217325 := by decide +kernel
theorem noise_83_5_bits : noiseBitsOfOut 13277290101518217325 = 0x3fdc2134802866d8 := by decide +kernel
theorem noise_83_6_key : noiseKey "kick-click" 6 = 17466507498146786764 := by decide +kernel
theorem noise_83_6_out : splitmix64 17466507498146786764 = 14646054318755411279 := by decide +kernel
theorem noise_83_6_bits : noiseBitsOfOut 14646054318755411279 = 0x3fe2d04f33819f7e := by decide +kernel
theorem noise_83_7_key : noiseKey "kick-click" 7 = 17541227791585119671 := by decide +kernel
theorem noise_83_7_out : splitmix64 17541227791585119671 = 4813187330170674844 := by decide +kernel
theorem noise_83_7_bits : noiseBitsOfOut 4813187330170674844 = 0xbfde9a0fcff1e210 := by decide +kernel
theorem noise_84_0_key : noiseKey "snare" 0 = 12946343628495346263 := by decide +kernel
theorem noise_84_0_out : splitmix64 12946343628495346263 = 2708203885605296638 := by decide +kernel
theorem noise_84_0_bits : noiseBitsOfOut 2708203885605296638 = 0xbfe69aa1ba8aca7a := by decide +kernel
theorem noise_84_1_key : noiseKey "snare" 1 = 9364022127782096002 := by decide +kernel
theorem noise_84_1_out : splitmix64 9364022127782096002 = 3681017551569361890 := by decide +kernel
theorem noise_84_1_bits : noiseBitsOfOut 3681017551569361890 = 0xbfe33a99828aaeb9 := by decide +kernel
theorem noise_84_2_key : noiseKey "snare" 2 = 2704555985273361009 := by decide +kernel
theorem noise_84_2_out : splitmix64 2704555985273361009 = 4917921371880839014 := by decide +kernel
theorem noise_84_2_bits : noiseBitsOfOut 4917921371880839014 = 0xbfdde00448c5ded4 := by decide +kernel
theorem noise_84_3_key : noiseKey "snare" 3 = 4185532912994999332 := by decide +kernel
theorem noise_84_3_out : splitmix64 4185532912994999332 = 5862142795974006635 := by decide +kernel
theorem noise_84_3_bits : noiseBitsOfOut 5862142795974006635 = 0xbfd752be17cfad54 := by decide +kernel
theorem noise_84_4_key : noiseKey "snare" 4 = 15290302756488202539 := by decide +kernel
theorem noise_84_4_out : splitmix64 15290302756488202539 = 10453581110190827357 := by decide +kernel
theorem noise_84_4_bits : noiseBitsOfOut 10453581110190827357 = 0x3fc11294a25fa868 := by decide +kernel
theorem noise_84_5_key : noiseKey "snare" 5 = 5729597892279332662 := by decide +kernel
theorem noise_84_5_out : splitmix64 5729597892279332662 = 11989810051263210315 := by decide +kernel
theorem noise_84_5_bits : noiseBitsOfOut 11989810051263210315 = 0x3fd3322e5bc2f9d0 := by decide +kernel
theorem noise_84_6_key : noiseKey "snare" 6 = 7467786887366787765 := by decide +kernel
theorem noise_84_6_out : splitmix64 7467786887366787765 = 11641547664517446510 := by decide +kernel
theorem noise_84_6_bits : noiseBitsOfOut 11641547664517446510 = 0x3fd0c78af5edab9c := by decide +kernel
theorem noise_84_7_key : noiseKey "snare" 7 = 10203527854999255640 := by decide +kernel
theorem noise_84_7_out : splitmix64 10203527854999255640 = 4363490178895338722 := by decide +kernel
theorem noise_84_7_bits : noiseBitsOfOut 4363490178895338722 = 0xbfe0dc7133448269 := by decide +kernel
theorem noise_85_0_key : noiseKey "hat" 0 = 12606923445487422547 := by decide +kernel
theorem noise_85_0_out : splitmix64 12606923445487422547 = 17040301273764626242 := by decide +kernel
theorem noise_85_0_bits : noiseBitsOfOut 17040301273764626242 = 0x3feb1ed3ee681820 := by decide +kernel
theorem noise_85_1_key : noiseKey "hat" 1 = 8552177738194990570 := by decide +kernel
theorem noise_85_1_out : splitmix64 8552177738194990570 = 11036983019463291455 := by decide +kernel
theorem noise_85_1_bits : noiseBitsOfOut 11036983019463291455 = 0x3fc92b3d946b20b0 := by decide +kernel
theorem noise_85_2_key : noiseKey "hat" 2 = 12358952426624498257 := by decide +kernel
theorem noise_85_2_out : splitmix64 12358952426624498257 = 4299755477057016401 := by decide +kernel
theorem noise_85_2_bits : noiseBitsOfOut 4299755477057016401 = 0xbfe1150ccb2e3e7c := by decide +kernel
theorem noise_85_3_key : noiseKey "hat" 3 = 1780697256161370256 := by decide +kernel
theorem noise_85_3_out : splitmix64 1780697256161370256 = 15556622834910443300 := by decide +kernel
theorem noise_85_3_bits : noiseBitsOfOut 15556622834910443300 = 0x3fe5f90e82eee44e := by decide +kernel
theorem noise_85_4_key : noiseKey "hat" 4 = 6712628573405605831 := by decide +kernel
theorem noise_85_4_out : splitmix64 6712628573405605831 = 4008505917493343855 := by decide +kernel
theorem noise_85_4_bits : noiseBitsOfOut 4008505917493343855 = 0xbfe217bb46c8f082 := by decide +kernel
theorem noise_85_5_key : noiseKey "hat" 5 = 5123292136216393774 := by decide +kernel
theorem noise_85_5_out : splitmix64 5123292136216393774 = 15576325530533134398 := by decide +kernel
theorem noise_85_5_bits : noiseBitsOfOut 15576325530533134398 = 0x3fe60a8e62c251a0 := by decide +kernel
theorem noise_85_6_key : noiseKey "hat" 6 = 15197090794021305653 := by decide +kernel
theorem noise_85_6_out : splitmix64 15197090794021305653 = 12522425150617932330 := by decide +kernel
theorem noise_85_6_bits : noiseBitsOfOut 12522425150617932330 = 0x3fd6e44ba9f67fe8 := by decide +kernel
theorem noise_85_7_key : noiseKey "hat" 7 = 18291501920128455364 := by decide +kernel
theorem noise_85_7_out : splitmix64 18291501920128455364 = 15161440175345698449 := by decide +kernel
theorem noise_85_7_bits : noiseBitsOfOut 15161440175345698449 = 0x3fe49a1060afb442 := by decide +kernel
theorem noise_86_0_key : noiseKey "clap" 0 = 15397128623213183397 := by decide +kernel
theorem noise_86_0_out : splitmix64 15397128623213183397 = 13800784395528873836 := by decide +kernel
theorem noise_86_0_bits : noiseBitsOfOut 13800784395528873836 = 0x3fdfc31e24dea054 := by decide +kernel
theorem noise_86_1_key : noiseKey "clap" 1 = 15207513820784737330 := by decide +kernel
theorem noise_86_1_out : splitmix64 15207513820784737330 = 11799470361974286156 := by decide +kernel
theorem noise_86_1_bits : noiseBitsOfOut 11799470361974286156 = 0x3fd1e011e3939d14 := by decide +kernel
theorem noise_86_2_key : noiseKey "clap" 2 = 6517245782479580563 := by decide +kernel
theorem noise_86_2_out : splitmix64 6517245782479580563 = 11592944544076193383 := by decide +kernel
theorem noise_86_2_bits : noiseBitsOfOut 11592944544076193383 = 0x3fd07134d2053b70 := by decide +kernel
theorem noise_86_3_key : noiseKey "clap" 3 = 1249845198913559496 := by decide +kernel
theorem noise_86_3_out : splitmix64 1249845198913559496 = 6790643279675575386 := by decide +kernel
theorem noise_86_3_bits : noiseBitsOfOut 6790643279675575386 = 0xbfd0e164f52f459a := by decide +kernel
theorem noise_86_4_key : noiseKey "clap" 4 = 17829101139857296617 := by decide +kernel
theorem noise_86_4_out : splitmix64 17829101139857296617 = 9013968774224448498 := by decide +kernel
theorem noise_86_4_bits : noiseBitsOfOut 9013968774224448498 = 0xbf973f99435f1ba0 := by decide +kernel
theorem noise_86_5_key : noiseKey "clap" 5 = 1795039206625499334 := by decide +kernel
theorem noise_86_5_out : splitmix64 1795039206625499334 = 4376130418523372685 := by decide +kernel
theorem noise_86_5_bits : noiseBitsOfOut 4376130418523372685 = 0xbfe0d137247c4881 := by decide +kernel
theorem noise_86_6_key : noiseKey "clap" 6 = 15697636122230672535 := by decide +kernel
theorem noise_86_6_out : splitmix64 15697636122230672535 = 12932505633848647880 := by decide +kernel
theorem noise_86_6_bits : noiseBitsOfOut 12932505633848647880 = 0x3fd9bcbead64a4b0 := by decide +kernel
theorem noise_86_7_key : noiseKey "clap" 7 = 76111485216008940 := by decide +kernel
theorem noise_86_7_out : splitmix64 76111485216008940 = 15994444750496922817 := by decide +kernel
theorem noise_86_7_bits : noiseBitsOfOut 15994444750496922817 = 0x3fe77debb0893ec0 := by decide +kernel
theorem probe_0 : fltBitsOfOut 18446744073709551615 = 0x3ff0000000000000 := by decide +kernel
theorem probe_1 : fltBitsOfOut 9007199254740993 = 0x3f40000000000000 := by decide +kernel
theorem probe_2 : fltBitsOfOut 6148914691236517205 = 0x3fd5555555555555 := by decide +kernel

/-! ## Float agreement tests (opaque Float, #eval only) -/
def fltTable : List (UInt64 × UInt64) := [(12757181382089057799, 0x3fe621539c0cd4a5),
  (467639867812890905, 0x3f99f58fedaf3124),
  (16451063564222362334, 0x3fec89bd64ce5b0e),
  (11243530964749613534, 0x3fe381217aeee02f),
  (17889463194238400601, 0x3fef08847fc46027),
  (3329825183073691434, 0x3fc71af52e50a0f4),
  (10418166507254056483, 0x3fe2129867b27801),
  (4262026829083468677, 0x3fcd92e168f15a32),
  (6255470360344140550, 0x3fd5b3f94996d307),
  (4406150150260045110, 0x3fce92e51bc2a45b),
  (16393747132694783303, 0x3fec70494519bb3d),
  (12713720907164796623, 0x3fe60e06b9c1a079),
  (663972912476720797, 0x3fa26dcfb129501d),
  (2339962062033943810, 0x3fc03c9b8c83b63f),
  (5770775617285411770, 0x3fd4057a7656b215),
  (3390881933790995449, 0x3fc7876a93997372),
  (4086868642287052498, 0x3fcc5bbcb04199f3),
  (18105136213321633839, 0x3fef684baebfa31a),
  (5510905795640608826, 0x3fd31eaae7e0f332),
  (8556394189718001347, 0x3fddaf9acba316e6),
  (9965958867797129579, 0x3fe149c6593a2677),
  (5373665931711646291, 0x3fd2a4c62c6a52a3),
  (12462952158586089905, 0x3fe59ea99e9d26c6),
  (12892727376979303825, 0x3fe65d8567b459c4),
  (6058400220100580244, 0x3fd504f0b9b08f95),
  (2766249899105893150, 0x3fc331d8d042155e),
  (16741192582307531419, 0x3fed0a953e8f1b09),
  (10070074555902414558, 0x3fe17802ee8fc86a),
  (1985904175011501217, 0x3fbb8f59534d8695),
  (17623461022340051418, 0x3fee92638da7b0ae),
  (18295920571732716982, 0x3fefbd055a5e9c7e),
  (16361030053194388779, 0x3fec61c1c4c4048b),
  (6713537841450575239, 0x3fd74ad1c6347379),
  (4936374817134311743, 0x3fd12061af7f5e5e),
  (9823027550008932344, 0x3fe10a4cf063250f),
  (8226272594980673756, 0x3fdc8a65d5ca56a6),
  (8413439981506867135, 0x3fdd30a2c58f324c),
  (2515536648523032331, 0x3fc1747da0815894),
  (18364650057183304505, 0x3fefdb8afda95893),
  (2284487120611317957, 0x3fbfb420eeb5fe19),
  (12846207557701622176, 0x3fe648dcb6c577cb),
  (6411241653308588440, 0x3fd63e539430a960),
  (10573257400230596023, 0x3fe2577832c0707a),
  (475549350226305700, 0x3f9a65f67535a08b),
  (5931852765154887047, 0x3fd4948b2a57be71),
  (120161086863855059, 0x3f7aae5df32584dd),
  (4043171810137051012, 0x3fcc0e1dab7a8bc8),
  (7560668312444837038, 0x3fda3b38f168f4e7),
  (12846207557701622176, 0x3fe648dcb6c577cb),
  (6411241653308588440, 0x3fd63e539430a960),
  (10573257400230596023, 0x3fe2577832c0707a),
  (475549350226305700, 0x3f9a65f67535a08b),
  (5931852765154887047, 0x3fd4948b2a57be71),
  (120161086863855059, 0x3f7aae5df32584dd),
  (4043171810137051012, 0x3fcc0e1dab7a8bc8),
  (7560668312444837038, 0x3fda3b38f168f4e7),
  (8238595112922386140, 0x3fdc9557a6bc8084),
  (4863424868985589985, 0x3fd0df96c9a6ebf4),
  (5514382935022184792, 0x3fd321c184075e4a),
  (13910272450057628076, 0x3fe82166e2fe52cd),
  (14142616848627420883, 0x3fe8889562fa1c8a),
  (4458133278213669396, 0x3fceef3c4c54a301),
  (11035409188825569172, 0x3fe324b4c60d60f3),
  (4708991831262369427, 0x3fd0566cc7c22706)]
def noiseTable : List (String × Nat × UInt64) := [("kick-click", 0, 0xbfe465033ac017f4),
  ("kick-click", 1, 0xbfddd575ce73fc5e),
  ("kick-click", 2, 0xbfbae8833765f478),
  ("kick-click", 3, 0x3fe551e702026a14),
  ("kick-click", 4, 0xbfded0888553777c),
  ("kick-click", 5, 0x3fdc2134802866d8),
  ("kick-click", 6, 0x3fe2d04f33819f7e),
  ("kick-click", 7, 0xbfde9a0fcff1e210),
  ("snare", 0, 0xbfe69aa1ba8aca7a),
  ("snare", 1, 0xbfe33a99828aaeb9),
  ("snare", 2, 0xbfdde00448c5ded4),
  ("snare", 3, 0xbfd752be17cfad54),
  ("snare", 4, 0x3fc11294a25fa868),
  ("snare", 5, 0x3fd3322e5bc2f9d0),
  ("snare", 6, 0x3fd0c78af5edab9c),
  ("snare", 7, 0xbfe0dc7133448269),
  ("hat", 0, 0x3feb1ed3ee681820),
  ("hat", 1, 0x3fc92b3d946b20b0),
  ("hat", 2, 0xbfe1150ccb2e3e7c),
  ("hat", 3, 0x3fe5f90e82eee44e),
  ("hat", 4, 0xbfe217bb46c8f082),
  ("hat", 5, 0x3fe60a8e62c251a0),
  ("hat", 6, 0x3fd6e44ba9f67fe8),
  ("hat", 7, 0x3fe49a1060afb442),
  ("clap", 0, 0x3fdfc31e24dea054),
  ("clap", 1, 0x3fd1e011e3939d14),
  ("clap", 2, 0x3fd07134d2053b70),
  ("clap", 3, 0xbfd0e164f52f459a),
  ("clap", 4, 0xbf973f99435f1ba0),
  ("clap", 5, 0xbfe0d137247c4881),
  ("clap", 6, 0x3fd9bcbead64a4b0),
  ("clap", 7, 0x3fe77debb0893ec0)]
def probeTable : List (UInt64 × UInt64) := [(18446744073709551615, 0x3ff0000000000000), (9007199254740993, 0x3f40000000000000), (6148914691236517205, 0x3fd5555555555555)]

def fltOk := fltTable.filter fun (sm, b) => (fltOfOut sm).toBits == b
def fltModelOk := fltTable.filter fun (sm, b) => fltBitsOfOut sm == b
def fltFloatVsModel := fltTable.filter fun (sm, _) => (fltOfOut sm).toBits == fltBitsOfOut sm
def noiseOk := noiseTable.filter fun (tag, i, b) => (noiseAt tag i).toBits == b
def noiseFloatVsModel := noiseTable.filter fun (tag, i, _) => (noiseAt tag i).toBits == noiseBitsOfOut (splitmix64 (noiseKey tag i))
def probeOk := probeTable.filter fun (x, b) => (fltOfOut x).toBits == b
#eval IO.println s!"flt Float==golden: {fltOk.length}/{fltTable.length}"
#eval IO.println s!"flt model==golden: {fltModelOk.length}/{fltTable.length}"
#eval IO.println s!"flt Float==model:  {fltFloatVsModel.length}/{fltTable.length}"
#eval IO.println s!"noise Float==golden: {noiseOk.length}/{noiseTable.length}"
#eval IO.println s!"noise Float==model:  {noiseFloatVsModel.length}/{noiseTable.length}"
#eval IO.println s!"probes Float==golden: {probeOk.length}/{probeTable.length}"
#eval IO.println s!"fltOfOut (2^64-1) == 1.0: {fltOfOut 18446744073709551615 == 1.0}  bits={(fltOfOut 18446744073709551615).toBits}"
#eval IO.println s!"noise snare 8 == (noise snare 64).take 8 (Float): {(noise "snare" 8).map Float.toBits == ((noise "snare" 64).take 8).map Float.toBits}"

