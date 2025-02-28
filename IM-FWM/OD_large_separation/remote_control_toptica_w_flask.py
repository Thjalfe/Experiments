# %%
import requests


class DLCControlClient:
    def __init__(self, remote_ip, cert_path):
        self.remote_ip = remote_ip
        if cert_path:
            self.base_url = f"https://{self.remote_ip}:5000"
        else:
            self.base_url = f"http://{self.remote_ip}:5000"
        self.cert_path = cert_path  # Path to the server certificate

    def _send_request(self, method, endpoint, data=None):
        """Helper method to send requests to the Flask server."""
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, verify=self.cert_path)
            elif method == "POST":
                response = requests.post(url, json=data, verify=self.cert_path)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"Error: {response.status_code}, {response.text}",
                }
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}

    def open_connection(self):
        """Open the connection to the DLC laser."""
        return self._send_request("POST", "/open")

    def close_connection(self):
        """Close the connection to the DLC laser."""
        return self._send_request("POST", "/close")

    def get_limits(self):
        """Get the laser parameters' limits."""
        return self._send_request("GET", "/get_limits")

    def get_wavelength(self):
        """Get the current wavelength."""
        return self._send_request("GET", "/get_wavelength")

    def set_wavelength(self, wavelength):
        """Set the wavelength of the laser."""
        return self._send_request("POST", "/set_wavelength", {"wavelength": wavelength})

    def get_current(self):
        """Get the current value."""
        return self._send_request("GET", "/get_current")

    def set_current(self, current):
        """Set the current of the laser."""
        return self._send_request("POST", "/set_current", {"current": current})

    def get_emission(self):
        """Get the emission status."""
        return self._send_request("GET", "/get_emission")

    def enable_emission(self):
        """Enable the emission."""
        return self._send_request("POST", "/enable_emission")

    def disable_emission(self):
        """Disable the emission."""
        return self._send_request("POST", "/disable_emission")


if __name__ == "__main__":
    remote_ip = "10.51.37.182"  # Replace with the actual IP of the remote PC
    cert_path = r"C:\Users\FTNK-FOD\Desktop\Thjalfe\ssl\server.crt"  # Path to the server certificate
    # cert_path = False
    dlc_client = DLCControlClient(remote_ip, cert_path)
# dlc_client.open_connection()
