import React from 'react'
import Header3 from './Header3'
import ScrollButton from './ScrollButton'
import Button from './Button'
import { Link } from 'react-router-dom'

function Heroe({display, children, subheading, scrollButton, linkButton}) {
    return (
        <div className="d-flex p-5 justify-content-center align-items-center text-center bg-image rounded-3 bg-image h-100 w-100">
            <div className='mask'>
                <div className=" card-title h-100">
                    <div className="dark-green">
                        <Header3 className={"mb-3 stroke " + display}>{children}</Header3>
                        {
                            subheading ? <h4 className="mb-3">{subheading}</h4> : ""
                        }
                        {
                            scrollButton ? <ScrollButton className="btn-lg" theme={"outline-danger"} href={"#pasteData"}>{scrollButton}</ScrollButton> : ""
                        }
                        {
                            linkButton ? <Link to={"/single_customer"}><Button theme={"primary"} className={""}>{linkButton}</Button></Link> : ""
                        }
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Heroe