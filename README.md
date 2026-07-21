# CSTR Sensitivity

Local feedback gains from one solve: parametric sensitivity on a CSTR

**Live demo:** https://cstr-sensitivity.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines. Custom domain wired through Cloudflare DNS.

- **Machine**: `shared-cpu-1x` · 1 GB RAM · single region (`ord`) · `min_machines_running=0` (auto-stops on idle).
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7. Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

## Files

- `app.py`: Streamlit UI and computation
- `schematic.png`: the CSTR schematic shown in the app and the notebook
- `requirements.txt`: Python deps
- `favicon.png`: Griffith PSE blackletter G favicon
- `Dockerfile`, `fly.toml`, `.dockerignore`: Fly.io production image config
- `.streamlit/config.toml`: Streamlit defaults baked into the image
- `.github/workflows/deploy.yml`: auto-deploy pipeline

## References

- G. A. Hicks and W. H. Ray, "Approximation methods for optimal control
  synthesis," Can. J. Chem. Eng. 49 (1971) 522-528.
  [DOI](https://doi.org/10.1002/cjce.5450490416)
- R. Huang, S. C. Patwardhan, and L. T. Biegler, "Robust stability of
  nonlinear model predictive control based on extended Kalman filter,"
  J. Process Control 22 (2012) 82-89 (the dimensionless form used here).
  [DOI](https://doi.org/10.1016/j.jprocont.2011.10.006)
- V. M. Zavala and L. T. Biegler, "The advanced-step NMPC controller:
  optimality, stability and robustness," Automatica 45 (2009) 86-93.
  [DOI](https://doi.org/10.1016/j.automatica.2008.06.011)
- H. Pirnay, R. Lopez-Negrete, and L. T. Biegler, "Optimal sensitivity
  based on IPOPT," Math. Program. Comput. 4 (2012) 307-331 (sIPOPT, the
  style of sensitivity used here).
  [DOI](https://doi.org/10.1007/s12532-012-0043-2)
