import unittest
import numpy as np
from gate import fuse
from enhance import metrics

class GateTests(unittest.TestCase):
    def test_direction_and_range(self):
        dark=np.full((8,9,3),.02,dtype=np.float32);bright=np.full_like(dark,.8);enh=np.ones_like(dark)*2
        d,ad=fuse(dark,enh,global_map=True);b,ab=fuse(bright,enh,global_map=True)
        self.assertGreater(float(ad.mean()),.97);self.assertLess(float(ab.mean()),1e-6)
        self.assertTrue(np.all(d<=1));self.assertTrue(np.all(b>=bright))
    def test_global_scalar_equation(self):
        raw=np.random.default_rng(7).random((13,11,3),dtype=np.float32);enh=raw*.4+.6
        actual,a=fuse(raw,enh,global_map=True)
        mu=np.mean(raw@np.array([.2126,.7152,.0722],dtype=np.float32))
        alpha=1/(1+np.exp((mu-.18)/.04))
        np.testing.assert_allclose(actual,alpha*enh+(1-alpha)*raw,rtol=0,atol=2e-7)
        self.assertEqual(float(a.min()),float(a.max()))
    def test_identity_and_invalid_input(self):
        raw=np.full((8,8,3),.2,dtype=np.float32)
        y,_=fuse(raw,raw,global_map=True);np.testing.assert_allclose(y,raw,atol=1e-7)
        with self.assertRaises(ValueError):fuse(raw,raw[:4],global_map=True)
        with self.assertRaises(ValueError):fuse(raw,raw,temperature=0)
    def test_known_metric_and_perfect_match(self):
        ref=np.full((8,8,3),.5,dtype=np.float32);pred=np.full_like(ref,.25)
        m=metrics(pred,ref);self.assertAlmostEqual(m['psnr_db'],12.041199826559248,places=10)
        m=metrics(ref,ref);self.assertTrue(m['perfect_match']);self.assertAlmostEqual(m['mean_deltaE00'],0)

if __name__=='__main__':unittest.main()
