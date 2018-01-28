using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour {

    public Transform target;
    public int mouseButton = 0;
    public float distanceQuantity = 1.0f;

    Vector3 targetPos;
    

    void Start() {
        targetPos = target.position;
    }

    void Update() {
        // targetの移動量分、自分（カメラ）も移動する
        //transform.position += transform.position - targetPos;
        //targetPos = transform.position;
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        transform.position += distanceQuantity * scroll * transform.forward;

        // マウスの右クリックを押している間
        if(Input.GetMouseButton(mouseButton)) {
            // マウスの移動量
            float mouseInputX = Input.GetAxis("Mouse X");
            float mouseInputY = Input.GetAxis("Mouse Y");

            // targetの位置のY軸を中心に、回転（公転）する
            transform.RotateAround(targetPos, Vector3.up, mouseInputX * Time.deltaTime * 200f);
            // カメラの垂直移動（※角度制限なし、必要が無ければコメントアウト）
            transform.RotateAround(targetPos, transform.right, mouseInputY * Time.deltaTime * 200f);
        }
    }
}
